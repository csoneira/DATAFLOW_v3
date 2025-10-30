#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

from __future__ import annotations

"""
Created on Thu Jun 20 09:15:33 2024

@author: csoneira@ucm.es
"""



task_number = 2


# import glob
# import pandas as pd
# import random
# import os
# import sys

# # Pick a random file in "/home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_1/DONE/cleaned_<file>.h5"
# IN_PATH = glob.glob("/home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_1/DONE/cleaned_*.h5")[random.randint(0, len(glob.glob("/home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_1/DONE/cleaned_*.h5")) - 1)]
# KEY = "df"

# # Load dataframe
# working_df = pd.read_hdf(IN_PATH, key=KEY)
# print(f"✅ Cleaned dataframe reloaded from: {IN_PATH}")

# # --- Continue your calibration or analysis code here ---
# # e.g.:
# # run_calibration(working_df)


# # Take basename of IN_PATH without extension and witouth the 'cleaned_' prefix
# basename_no_ext = os.path.splitext(os.path.basename(IN_PATH))[0].replace("cleaned_", "")
# print(f"File basename (no extension): {basename_no_ext}")

# -----------------------------------------------------------------------------
# ------------------------------- Imports -------------------------------------
# -----------------------------------------------------------------------------

from datetime import datetime

# I want to chrono the execution time of the script
start_execution_time_counting = datetime.now()

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
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config.yaml")
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
home_path = config["home_path"]



def save_execution_metadata(home_dir: str, station_id: str, task_id: int, row: Dict[str, object]) -> Path:
    """Append the execution metadata row to the per-task CSV."""
    metadata_dir = (
        Path(home_dir)
        / "DATAFLOW_v3"
        / "STATIONS"
        / f"MINGO0{station_id}"
        / "STAGE_1"
        / "EVENT_DATA"
        / "STEP_1"
        / f"TASK_{task_id}"
        / "METADATA"
    )
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / "execution_metadata.csv"
    file_exists = metadata_path.exists()
    with metadata_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "filename_base",
                "execution_timestamp",
                "data_purity_percentage",
                "total_execution_time_minutes",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return metadata_path


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



print("Creating the necessary directories...")

date_execution = datetime.now().strftime("%y-%m-%d_%H.%M.%S")

# Define base working directory
home_directory = os.path.expanduser(f"~")
station_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}")
base_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/STAGE_1/EVENT_DATA")
raw_to_list_working_directory = os.path.join(base_directory, f"STEP_1/TASK_{task_number}")
if task_number == 1:
    raw_directory = "RAW"
else:
    raw_directory = f"STEP_1/TASK_{task_number - 1}/OUTPUT_FILES"
if task_number == 5:
    output_location = os.path.join(base_directory, "STEP_1_TO_2_OUTPUT")
else:
    output_location = os.path.join(raw_to_list_working_directory, "OUTPUT_FILES")
raw_working_directory = os.path.join(base_directory, raw_directory)

# /home/mingo/DATAFLOW_v3/STATIONS/MINGO01/STAGE_1/EVENT_DATA/STEP_1/TASK_1/OUTPUT_FILES
raw_working_directory = os.path.join(base_directory, "STEP_1/TASK_1/OUTPUT_FILES")

raw_to_list_working_directory = os.path.join(base_directory, "STEP_1/TASK_2/")

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
}

# Create ALL directories if they don't already exist
for directory in base_directories.values():
    os.makedirs(directory, exist_ok=True)

csv_path = os.path.join(base_directory, "raw_to_list_metadata.csv")
status_csv_path = os.path.join(base_directory, "raw_to_list_status.csv")
status_timestamp = append_status_row(status_csv_path)

# Move files from RAW to RAW_TO_LIST/RAW_TO_LIST_FILES/UNPROCESSED,
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


# Files to move: in RAW but not in UNPROCESSED, PROCESSING, or COMPLETED
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
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config.yaml")
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
home_path = config["home_path"]


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
create_very_essential_plots = config["create_very_essential_plots"]
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
create_plots_fast = config["create_plots_fast"]
create_plots_debug = config["create_plots_debug"]

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
# time_sum_reference = np.array([
#     [0.0, -0.3886308, -0.53020947, 0.33711737],
#     [-0.80494094, -0.68836069, -2.01289387, -1.13481931],
#     [-0.23899338, -0.51373738, 0.50845317, 0.11685095],
#     [0.33586385, 1.08329847, 0.91410244, 0.58815813]
# ])


time_sum_reference = np.array([
    [0., 0.34735892,0.45303939,-0.36284147],
    [0.8596194, 0.82903959,1.95702226,1.23050699],
    [0.34898167, 0.72719093,-0.2700321 ,0.01139674],
    [0.08731539,-0.61900897,-0.39471968,-0.07661417]
])
time_calibration = False


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
# Take basename of IN_PATH without extension and witouth the 'calibrated_' prefix
basename_no_ext = the_filename.replace("cleaned_", "").replace(".h5", "")

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

if limit:
    print(f'Taking the first {limit_number} rows.')


# Read the data file into a DataFrame


import glob
import pandas as pd
import random
import os
import sys

KEY = "df"

# Load dataframe
working_df = pd.read_hdf(file_path, key=KEY)
print(f"✅ Cleaned dataframe reloaded from: {file_path}")


original_number_of_events = len(working_df)
print(f"Original number of events in the dataframe: {original_number_of_events}")


# --- Continue your calibration or analysis code here ---
# e.g.:
# run_calibration(working_df)


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
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config.yaml")
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
home_path = config["home_path"]

ITINERARY_FILE_PATH = Path(
    f"{home_path}/DATAFLOW_v3/MASTER/CONFIG_FILES/itineraries.csv"
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
create_very_essential_plots = config["create_very_essential_plots"]
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
create_plots_fast = config["create_plots_fast"]
create_plots_debug = config["create_plots_debug"]

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
# -----------------------------------------------------------------------------
# Function definition ---------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def calibrate_strip_T_diff(T_F, T_B, self_trigger_mode = False):
    
    if self_trigger_mode:
        T_left_side = T_F_left_pre_cal_ST
        T_right_side = T_F_right_pre_cal_ST
    else:
        T_left_side = T_F_left_pre_cal
        T_right_side = T_F_right_pre_cal
    
    cond = (T_F != 0) & (T_F > T_left_side) & (T_F < T_right_side) & (T_B != 0) & (T_B > T_left_side) & (T_B < T_right_side)
    
    # Front
    T_F = T_F[cond]
    counts, bin_edges = np.histogram(T_F, bins='auto')
    max_counts = np.max(counts)
    min_counts = np.min(counts[counts > 0])
    threshold = max_counts / 10**1.5
    indices_above_threshold = np.where(counts > threshold)[0]
    if indices_above_threshold.size > 0:
        min_bin_edge_F = bin_edges[indices_above_threshold[0]]
        max_bin_edge_F = bin_edges[indices_above_threshold[-1] + 1]  # +1 to get the upper edge of the last bin
        # print(f"Minimum bin edge: {min_bin_edge_F}")
        # print(f"Maximum bin edge: {max_bin_edge_F}")
    else:
        print("No bins have counts above the threshold, Front.")
        threshold = (min_counts + max_counts) / 2
        indices_above_threshold = np.where(counts > threshold)[0]
        min_bin_edge_F = bin_edges[indices_above_threshold[0]]
        max_bin_edge_F = bin_edges[indices_above_threshold[-1] + 1]
    
    # Back
    T_B = T_B[cond]
    counts, bin_edges = np.histogram(T_B, bins='auto')
    max_counts = np.max(counts)
    min_counts = np.min(counts[counts > 0])
    threshold = max_counts / 10**1.5
    indices_above_threshold = np.where(counts > threshold)[0]
    if indices_above_threshold.size > 0:
        min_bin_edge_B = bin_edges[indices_above_threshold[0]]
        max_bin_edge_B = bin_edges[indices_above_threshold[-1] + 1]  # +1 to get the upper edge of the last bin
        # print(f"Minimum bin edge: {min_bin_edge_B}")
        # print(f"Maximum bin edge: {max_bin_edge_B}")
    else:
        print("No bins have counts above the threshold, Back.")
        threshold = (min_counts + max_counts) / 2
        indices_above_threshold = np.where(counts > threshold)[0]
        min_bin_edge_B = bin_edges[indices_above_threshold[0]]
        max_bin_edge_B = bin_edges[indices_above_threshold[-1] + 1]
    
    cond = (T_F > min_bin_edge_F) & (T_F < max_bin_edge_F) & (T_B > min_bin_edge_B) & (T_B < max_bin_edge_B)
            
    T_F = T_F[cond]
    T_B = T_B[cond]
    
    # T_diff = ( T_F - T_B ) / 2
    T_diff = ( T_B - T_F ) / 2
    
    # ------------------------------------------------------------------------------
    
    
    T_rel_th = calibrate_strip_T_diff_T_rel_th
    abs_th = calibrate_strip_T_diff_T_abs_th

    # Apply mask to filter values within the threshold
    mask = (np.abs(T_diff) < T_diff_pre_cal_threshold)
    T_diff = T_diff[mask]
    
    # Remove zero values
    T_diff = T_diff[T_diff != 0]
    
    # Calculate histogram
    counts, bin_edges = np.histogram(T_diff, bins='auto')
    
    # Calculate the nunber of counts of the bin that has the most counts
    max_counts = np.max(counts)
    
    # Find bins with at least one count
    th = T_rel_th * max_counts
    if th < abs_th:
        th = abs_th
    non_empty_bins = counts >= th

    # Find the longest contiguous subset of non-empty bins
    max_length = 0
    current_length = 0
    start_index = 0
    temp_start = 0

    for i, is_non_empty in enumerate(non_empty_bins):
        if is_non_empty:
            if current_length == 0:
                temp_start = i
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                start_index = temp_start
                end_index = i
        else:
            current_length = 0
    
    plateau_left = bin_edges[start_index]
    plateau_right = bin_edges[end_index + 1]
    
    # Calculate the offset using the mean of the filtered values
    offset = ( plateau_left + plateau_right ) / 2
    
    return offset


def calibrate_strip_Q_pedestal(Q_ch, T_ch, Q_other, self_trigger_mode = False):
    
    translate_charge_cal = calibrate_strip_Q_pedestal_translate_charge_cal
    percentile = calibrate_strip_Q_pedestal_percentile
    
    rel_th = calibrate_strip_Q_pedestal_rel_th
    rel_th_cal = calibrate_strip_Q_pedestal_rel_th_cal
    abs_th = calibrate_strip_Q_pedestal_abs_th
    q_quantile = calibrate_strip_Q_pedestal_q_quantile # percentile
    
    # First let's tale good values of Time, we want to avoid outliers that might confuse the charge pedestal calibration
    
    if self_trigger_mode:
        T_left_side = T_F_left_pre_cal_ST
        T_right_side = T_F_right_pre_cal_ST
    else:
        T_left_side = T_F_left_pre_cal
        T_right_side = T_F_right_pre_cal
        
    cond = (T_ch != 0) & (T_ch > T_left_side) & (T_ch < T_right_side)
    T_ch = T_ch[cond]
    Q_ch = Q_ch[cond]
    Q_other = Q_other[cond]
    
    # Condition based on the charge difference: it cannot be too high
    Q_dif = Q_ch - Q_other
    
    cond = ( Q_dif > np.percentile(Q_dif, percentile) ) & ( Q_dif < np.percentile(Q_dif, 100 - percentile ) )
    T_ch = T_ch[cond]
    Q_ch = Q_ch[cond]
    
    counts, bin_edges = np.histogram(T_ch, bins='auto')
    max_counts = np.max(counts)
    min_counts = np.min(counts[counts > 0])
    threshold = max_counts / calibrate_strip_Q_pedestal_thr_factor
    
    indices_above_threshold = np.where(counts > threshold)[0]

    if indices_above_threshold.size > 0:
        min_bin_edge = bin_edges[indices_above_threshold[0]]
        max_bin_edge = bin_edges[indices_above_threshold[-1] + 1]  # +1 to get the upper edge of the last bin
    else:
        print("No bins have counts above the threshold; Q pedestal calibration.")
        threshold = (min_counts + max_counts) / calibrate_strip_Q_pedestal_thr_factor_2
        indices_above_threshold = np.where(counts > threshold)[0]
        min_bin_edge = bin_edges[indices_above_threshold[0]]
        max_bin_edge = bin_edges[indices_above_threshold[-1] + 1]
    
    Q_ch = Q_ch[(T_ch > min_bin_edge) & (T_ch < max_bin_edge)]
    
    # First take the values that are not zero
    Q_ch = Q_ch[Q_ch != 0]
    
    # Remove the values that are not in (50,500)
    Q_ch = Q_ch[(Q_ch > Q_left_side) & (Q_ch < Q_right_side)]
    
    # Quantile filtering
    Q_ch = Q_ch[Q_ch > np.percentile(Q_ch, q_quantile)]
    
    # Calculate histogram
    counts, bin_edges = np.histogram(Q_ch, bins='auto')
    
    # Calculate the nunber of counts of the bin that has the most counts
    max_counts = np.max(counts)
    counts = counts[counts < max_counts]
    max_counts = np.max(counts)
    
    # Find bins with at least one count
    th = rel_th * max_counts
    if th < abs_th:
        th = abs_th
    non_empty_bins = counts >= th

    # Find the longest contiguous subset of non-empty bins
    max_length = 0
    current_length = 0
    start_index = 0
    temp_start = 0

    for i, is_non_empty in enumerate(non_empty_bins):
        if is_non_empty:
            if current_length == 0:
                temp_start = i
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                start_index = temp_start
        else:
            current_length = 0

    # Get the first bin edge of the longest subset
    offset = bin_edges[start_index]
    
    # Second part --------------------------------------------------------------
    Q_ch_cal = Q_ch - offset
    
    # Remove values outside the range (-2, 2)
    Q_ch_cal = Q_ch_cal[(Q_ch_cal > pedestal_left) & (Q_ch_cal < pedestal_right)]
    
    # Calculate histogram
    counts, bin_edges = np.histogram(Q_ch_cal, bins='auto')
    
    # Find the bin with the most counts
    max_counts = np.max(counts)
    max_bin_index = np.argmax(counts)
    
    # Calculate the threshold
    threshold = rel_th_cal * max_counts
    
    # Start from the bin with the most counts and move left
    offset_bin_index = max_bin_index
    while offset_bin_index > 0 and counts[offset_bin_index] >= threshold:
        offset_bin_index -= 1
    
    # Determine the X value (left edge) of the bin where the threshold is crossed
    offset_cal = bin_edges[offset_bin_index]
    
    pedestal = offset + offset_cal
    pedestal = offset
    
    if translate_charge_cal:
        pedestal = pedestal - translate_charge_cal
        
    return pedestal


enumerate = builtins.enumerate
def polynomial(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))


def scatter_2d_and_fit_new(xdat, ydat, title, x_label, y_label, name_of_file):
    global fig_idx
    
    ydat_translated = ydat

    xdat_plot = xdat[(xdat < distance_sum_charges_plot) & (xdat > -distance_sum_charges_plot) & (ydat_translated < distance_sum_charges_plot) & (ydat_translated > -distance_sum_charges_plot)]
    ydat_plot = ydat_translated[(xdat < distance_sum_charges_plot) & (xdat > -distance_sum_charges_plot) & (ydat_translated < distance_sum_charges_plot) & (ydat_translated > -distance_sum_charges_plot)]
    xdat_pre_fit = xdat[(xdat < distance_sum_charges_right_fit) & (xdat > distance_sum_charges_left_fit) & (ydat_translated < distance_diff_charges_up_fit) & (ydat_translated > distance_diff_charges_low_fit)]
    ydat_pre_fit = ydat_translated[(xdat < distance_sum_charges_right_fit) & (xdat > distance_sum_charges_left_fit) & (ydat_translated < distance_diff_charges_up_fit) & (ydat_translated > distance_diff_charges_low_fit)]
    
    # Fit a polynomial of specified degree using curve_fit
    initial_guess = [1] * (degree_of_polynomial + 1)
    coeffs, _ = curve_fit(polynomial, xdat_pre_fit, ydat_pre_fit, p0=initial_guess)
    y_pre_fit = polynomial(xdat_pre_fit, *coeffs)
    
    # Filter data for fitting based on residues
    threshold = front_back_fit_threshold  # Set your desired threshold here
    residues = np.abs(ydat_pre_fit - y_pre_fit)  # Calculate residues
    xdat_fit = xdat_pre_fit[residues < threshold]
    ydat_fit = ydat_pre_fit[residues < threshold]
    
    # Perform fit on filtered data
    coeffs, _ = curve_fit(polynomial, xdat_fit, ydat_fit, p0=initial_guess)
    
    y_mean = np.mean(ydat_fit)
    y_check = polynomial(xdat_fit, *coeffs)
    ss_res = np.sum((ydat_fit - y_check)**2)
    ss_tot = np.sum((ydat_fit - y_mean)**2)
    r_squared = 1 - (ss_res / ss_tot)
    if r_squared < 0.5:
        print(f"---> R**2 in {name_of_file[0:4]}: {r_squared:.2g}")
    
    # if create_plots or create_essential_plots:
    if create_plots:
        x_fit = np.linspace(min(xdat_fit), max(xdat_fit), 100)
        y_fit = polynomial(x_fit, *coeffs)
        x_final = xdat_plot
        y_final = ydat_plot - polynomial(xdat_plot, *coeffs)
        plt.close()
        
        if article_format:
            ww = (10.84, 4) # (16,6) was very nice
        else:
            ww = (13.33, 5)
            
        plt.figure(figsize=ww)  # Use plt.subplots() to create figure and axis    
        plt.scatter(xdat_plot, ydat_plot, s=1, label="Original data points")
        # plt.scatter(xdat_pre_fit, ydat_pre_fit, s=1, color="magenta", label="Points for prefitting")
        plt.scatter(xdat_fit, ydat_fit, s=1, color="orange", label="Points for fitting")
        plt.scatter(x_final, y_final, s=1, color="green", label="Calibrated points")
        plt.plot(x_fit, y_fit, 'r-', label='Polynomial Fit: ' + ' '.join([f'a{i}={coeff:.2g}' for i, coeff in enumerate(coeffs[::-1])]))
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim([scatter_2d_and_fit_new_xlim_left, scatter_2d_and_fit_new_xlim_right])
        plt.ylim([scatter_2d_and_fit_new_ylim_bottom, scatter_2d_and_fit_new_ylim_top])
        plt.grid()
        plt.legend(markerscale=5)  # Increase marker scale by 5 times
        plt.tight_layout()
        if save_plots:
            name_of_file = 'charge_diff_vs_charge_sum_cal'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()
    return coeffs


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
    

def summary(vector):
    global coincidence_window_cal_ns
    quantile_left = CRT_gaussian_fit_quantile * 100
    quantile_right = 100 - CRT_gaussian_fit_quantile * 100
    
    vector = np.array(vector)  # Convert list to NumPy array
    cond = (vector > -coincidence_window_cal_ns) & (vector < coincidence_window_cal_ns)  # This should result in a boolean array
    vector = vector[cond]
    
    if len(vector) < 100:
        return np.nan
    try:
        percentile_left = np.percentile(vector, quantile_left)
        percentile_right = np.percentile(vector, quantile_right)
    except IndexError:
        print("Gave issue with:")
        print(vector)
        return np.nan
    vector = [x for x in vector if percentile_left <= x <= percentile_right]
    if len(vector) == 0:
        return np.nan
    mu, std = norm.fit(vector)
    return mu


def hist_1d(vdat, bin_number, title, axis_label, name_of_file):
    global fig_idx, coincidence_window_cal_ns
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    vdat = np.array(vdat)  # Convert list to NumPy array
    cond = (vdat > -coincidence_window_cal_ns) & (vdat < coincidence_window_cal_ns)  # This should result in a boolean array
    vdat = vdat[cond]
    counts, bins, _ = ax.hist(vdat, bins=bin_number, alpha=0.5, color="red", label=f"All hits, {len(vdat)} events", density=False)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    h1_q = CRT_gaussian_fit_quantile
    lower_bound = np.quantile(vdat, h1_q)
    upper_bound = np.quantile(vdat, 1 - h1_q)
    cond = (vdat > lower_bound) & (vdat < upper_bound)  # This should result in a boolean array
    vdat = vdat[cond]
    mu, std = norm.fit(vdat)
    p = norm.pdf(bin_centers, mu, std) * len(vdat) * (bins[1] - bins[0])  # Scale to match histogram
    label_plot = f'Gaussian fit:\n    $\\mu={mu:.2g}$,\n    $\\sigma={std:.2g}$\n    CRT$={std/np.sqrt(2)*1000:.3g}$ ps'
    ax.plot(bin_centers, p, 'k', linewidth=2, label=label_plot)
    ax.legend()
    ax.set_title(title)
    plt.xlabel(axis_label)
    plt.ylabel("Counts")
    plt.tight_layout()
    if save_plots:
        name_of_file = 'timing'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots: plt.show()
    plt.close()


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
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ------------------- TASK_2: calibrations ------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


print("--------------------------------------------------------------------------")
print("-------------------- Charge pedestal calibration -------------------------")
print("--------------------------------------------------------------------------")

charge_test = working_df.copy()
charge_test_copy = charge_test.copy()

# New pedestal calibration for charges ------------------------------------------------
QF_pedestal = []
for key in ['1', '2', '3', '4']:
    Q_F_cols = [f'Q{key}_F_{i+1}' for i in range(4)]
    Q_F = working_df[Q_F_cols].values
    
    Q_B_cols = [f'Q{key}_B_{i+1}' for i in range(4)]
    Q_B = working_df[Q_B_cols].values
    
    T_F_cols = [f'T{key}_F_{i+1}' for i in range(4)]
    T_F = working_df[T_F_cols].values
    
    QF_pedestal_component = [calibrate_strip_Q_pedestal(Q_F[:,i], T_F[:,i], Q_B[:,i]) for i in range(4)]
    QF_pedestal.append(QF_pedestal_component)
QF_pedestal = np.array(QF_pedestal)

QB_pedestal = []
for key in ['1', '2', '3', '4']:
    Q_F_cols = [f'Q{key}_F_{i+1}' for i in range(4)]
    Q_F = working_df[Q_F_cols].values
    
    Q_B_cols = [f'Q{key}_B_{i+1}' for i in range(4)]
    Q_B = working_df[Q_B_cols].values
    
    T_B_cols = [f'T{key}_B_{i+1}' for i in range(4)]
    T_B = working_df[T_B_cols].values
    
    QB_pedestal_component = [calibrate_strip_Q_pedestal(Q_B[:,i], T_B[:,i], Q_F[:,i]) for i in range(4)]
    QB_pedestal.append(QB_pedestal_component)
QB_pedestal = np.array(QB_pedestal)

print("\nFront Charge Pedestal:")
print(QF_pedestal)
print("\nBack Charge Pedestal:")
print(QB_pedestal,"\n")

for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    for j in range(4):
        mask = charge_test_copy[f'{key}_F_{j+1}'] != 0
        charge_test.loc[mask, f'{key}_F_{j+1}'] -= QF_pedestal[i][j]

for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    for j in range(4):
        mask = charge_test_copy[f'{key}_B_{j+1}'] != 0
        charge_test.loc[mask, f'{key}_B_{j+1}'] -= QB_pedestal[i][j]


# Plot histograms of all the pedestal substractions
if validate_charge_pedestal_calibration:
    # if create_plots or create_essential_plots:
    if create_plots:
        # Create the grand figure for Q values
        fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_Q = axes_Q.flatten()
        
        for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
            for j in range(4):
                col_F = f'{key}_F_{j+1}'
                col_B = f'{key}_B_{j+1}'
                y_F = charge_test[col_F]
                y_B = charge_test[col_B]
                
                # Plot histograms with Q-specific clipping and bins
                axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
                axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min) & (y_B < Q_clip_max)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
                axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
                axes_Q[i*4 + j].legend()
                
                if log_scale:
                    axes_Q[i*4 + j].set_yscale('log')  # For Q values

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Grand Figure for pedestal substracted values, mingo0{station}\n{start_time}", fontsize=16)
        
        if save_plots:
            final_filename = f'{fig_idx}_grand_figure_Q_pedestal.png'
            fig_idx += 1
            
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        
        if show_plots: plt.show()
        plt.close(fig_Q)
        
        
    if create_plots or create_essential_plots:
    # if create_plots:
        # ZOOOOOOOOOOOOOOOOOOOM ------------------------------------------------
        # Create the grand figure for Q values
        fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_Q = axes_Q.flatten()
        
        for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
            for j in range(4):
                col_F = f'{key}_F_{j+1}'
                col_B = f'{key}_B_{j+1}'
                y_F = charge_test[col_F]
                y_B = charge_test[col_B]
                
                Q_clip_min = pedestal_left
                Q_clip_max = pedestal_right
                
                # Plot histograms with Q-specific clipping and bins
                axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
                axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min) & (y_B < Q_clip_max)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
                axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
                axes_Q[i*4 + j].legend()
                # Show between -5 and 5
                axes_Q[i*4 + j].set_xlim([Q_clip_min, Q_clip_max])
        # Display a vertical green dashed, alpha = 0.5 line at 0
        for ax in axes_Q:
            ax.axvline(0, color='green', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Grand Figure for pedestal substracted values (zoom), mingo0{station}\n{start_time}", fontsize=16)
        
        if save_plots:
            final_filename = f'{fig_idx}_grand_figure_Q_pedestal_zoom.png'
            fig_idx += 1
            
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        
        if show_plots: plt.show()
        plt.close(fig_Q)
    
    
    if create_plots or create_essential_plots:
    # if create_plots:
        # ZOOOOOOOOOOOOOM ------------------------------------------------
        # Create the grand figure for Q values
        fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_Q = axes_Q.flatten()
        
        for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
            for j in range(4):
                col_F = f'{key}_F_{j+1}'
                col_B = f'{key}_B_{j+1}'
                y_F = charge_test[col_F]
                y_B = charge_test[col_B]
                
                Q_clip_min = pedestal_left * 2
                Q_clip_max = pedestal_right * 12
                
                # Plot histograms with Q-specific clipping and bins
                axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
                axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min) & (y_B < Q_clip_max)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
                axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
                axes_Q[i*4 + j].legend()
                # Show between -5 and 5
                axes_Q[i*4 + j].set_xlim([Q_clip_min, Q_clip_max])
        # Display a vertical green dashed, alpha = 0.5 line at 0
        for ax in axes_Q:
            ax.axvline(0, color='green', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Grand Figure for pedestal substracted values (zoom), mingo0{station}\n{start_time}", fontsize=16)
        
        if save_plots:
            final_filename = f'{fig_idx}_grand_figure_Q_pedestal_less_zoom.png'
            fig_idx += 1
            
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        
        if show_plots: plt.show()
        plt.close(fig_Q)


# ----------------------------------------------------------------------------------
# ----------------------- Charge calibration from ns to fC -------------------------
# ----------------------------------------------------------------------------------

if calibrate_charge_ns_to_fc:

    # Load calibration
    home_path = config["home_path"]
    tot_to_charge_cal_path = f"{home_path}/DATAFLOW_v3/MASTER/CONFIG_FILES/tot_to_charge_calibration.csv"
    FEE_calibration_df = pd.read_csv(tot_to_charge_cal_path)
    FEE_calibration = {
        "Width": FEE_calibration_df['Width'].tolist(),
        "Fast Charge": FEE_calibration_df['Fast_Charge'].tolist() }
    FEE_calibration = pd.DataFrame(FEE_calibration)

    width_table = FEE_calibration['Width'].to_numpy()
    fast_charge_table = FEE_calibration['Fast Charge'].to_numpy()
    # Create a cubic spline interpolator
    cs = CubicSpline(width_table, fast_charge_table, bc_type='natural')

    

    # --- Calibrate and store new columns in working_df ---
    for key in ['Q1', 'Q2', 'Q3', 'Q4']:
        for j in range(1, 5):
            for suffix in ['F', 'B']:
                col = f"{key}_{suffix}_{j}"
                if col in charge_test.columns:
                    col_fC = f"{col}_fC"
                    raw = charge_test[col]
                    mask = (raw != 0) & np.isfinite(raw)
                    charge_test[col_fC] = 0.0  # initialize
                    charge_test.loc[mask, col_fC] = interpolate_fast_charge(raw[mask])

    if create_plots:
        
        Q_clip_min = interpolate_fast_charge_Q_clip_min
        Q_clip_max = interpolate_fast_charge_Q_clip_max
        num_bins = interpolate_fast_charge_num_bins
        log_scale = interpolate_fast_charge_log_scale
        
        fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))
        axes_Q = axes_Q.flatten()

        for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
            for j in range(4):
                col_F = f'{key}_F_{j+1}_fC'
                col_B = f'{key}_B_{j+1}_fC'
                ax = axes_Q[i*4 + j]

                if col_F in charge_test.columns:
                    y_F = charge_test[col_F]
                    y_F = y_F[(y_F > Q_clip_min) & (y_F < Q_clip_max) & np.isfinite(y_F)]
                    ax.hist(y_F, bins=num_bins, alpha=0.5, label=f'{col_F}')

                if col_B in charge_test.columns:
                    y_B = charge_test[col_B]
                    y_B = y_B[(y_B > Q_clip_min) & (y_B < Q_clip_max) & np.isfinite(y_B)]
                    ax.hist(y_B, bins=num_bins, alpha=0.5, label=f'{col_B}')

                ax.set_title(f"{col_F} vs {col_B}")
                ax.set_xlabel('Charge [fC]')
                ax.legend()

                if log_scale:
                    ax.set_yscale('log')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Grand Figure for calibrated charge (fC), mingo0{station}\n{start_time}", fontsize=16)

        if save_plots:
            final_filename = f'{fig_idx}_grand_figure_Q_fC.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')

        if show_plots:
            plt.show()
        plt.close(fig_Q)


if self_trigger:
    print("--------------------------------------------------------------------------")
    print("---------------- SELF TRIGGER Charge pedestal calibration-----------------")
    print("--------------------------------------------------------------------------")

    charge_test = working_st_df.copy()
    charge_test_copy = charge_test.copy()

    # New pedestal calibration for charges ------------------------------------------------
    QF_pedestal_ST = []
    for key in ['1', '2', '3', '4']:
        Q_F_cols = [f'Q{key}_F_{i+1}' for i in range(4)]
        Q_F = working_st_df[Q_F_cols].values
    
        Q_B_cols = [f'Q{key}_B_{i+1}' for i in range(4)]
        Q_B = working_st_df[Q_B_cols].values
    
        T_F_cols = [f'T{key}_F_{i+1}' for i in range(4)]
        T_F = working_st_df[T_F_cols].values
    
        QF_pedestal_component = [calibrate_strip_Q_pedestal(Q_F[:,i], T_F[:,i], Q_B[:,i], self_trigger_mode = self_trigger) for i in range(4)]
        QF_pedestal_ST.append(QF_pedestal_component)
    QF_pedestal_ST = np.array(QF_pedestal_ST)

    QB_pedestal_ST = []
    for key in ['1', '2', '3', '4']:
        Q_F_cols = [f'Q{key}_F_{i+1}' for i in range(4)]
        Q_F = working_st_df[Q_F_cols].values
    
        Q_B_cols = [f'Q{key}_B_{i+1}' for i in range(4)]
        Q_B = working_st_df[Q_B_cols].values
    
        T_B_cols = [f'T{key}_B_{i+1}' for i in range(4)]
        T_B = working_st_df[T_B_cols].values
    
        QB_pedestal_component = [calibrate_strip_Q_pedestal(Q_B[:,i], T_B[:,i], Q_F[:,i], self_trigger_mode = self_trigger) for i in range(4)]
        QB_pedestal_ST.append(QB_pedestal_component)
    QB_pedestal_ST = np.array(QB_pedestal_ST)

    print("\nSELF TRIGGER Front Charge Pedestal:")
    print(QF_pedestal_ST)
    print("\nSELF TRIGGER Back Charge Pedestal:")
    print(QB_pedestal_ST,"\n")

    for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        for j in range(4):
            mask = charge_test_copy[f'{key}_F_{j+1}'] != 0
            charge_test.loc[mask, f'{key}_F_{j+1}'] -= QF_pedestal_ST[i][j]

    for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        for j in range(4):
            mask = charge_test_copy[f'{key}_B_{j+1}'] != 0
            charge_test.loc[mask, f'{key}_B_{j+1}'] -= QB_pedestal_ST[i][j]


    # Plot histograms of all the pedestal substractions
    validate_charge_pedestal_calibration = True
    if validate_charge_pedestal_calibration:
        # if create_plots or create_essential_plots:
        if create_plots:
            # Create the grand figure for Q values
            fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
            axes_Q = axes_Q.flatten()
        
            for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
                for j in range(4):
                    col_F = f'{key}_F_{j+1}'
                    col_B = f'{key}_B_{j+1}'
                    y_F = charge_test[col_F]
                    y_B = charge_test[col_B]
                
                    # Plot histograms with Q-specific clipping and bins
                    axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                        bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
                    axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min) & (y_B < Q_clip_max)], 
                                        bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
                    axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
                    axes_Q[i*4 + j].legend()
                
                    if log_scale:
                        axes_Q[i*4 + j].set_yscale('log')  # For Q values

            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.suptitle(f"Grand Figure for pedestal substracted values, mingo0{station}\n{start_time}", fontsize=16)
        
            if save_plots:
                final_filename = f'{fig_idx}_grand_figure_Q_pedestal_ST.png'
                fig_idx += 1
            
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
        
            if show_plots: plt.show()
            plt.close(fig_Q)
        
        
        if create_plots or create_essential_plots:
        # if create_plots:
            # ZOOOOOOOOOOOOOOOOOOOM ------------------------------------------------
            # Create the grand figure for Q values
            fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
            axes_Q = axes_Q.flatten()
        
            for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
                for j in range(4):
                    col_F = f'{key}_F_{j+1}'
                    col_B = f'{key}_B_{j+1}'
                    y_F = charge_test[col_F]
                    y_B = charge_test[col_B]
                
                    Q_clip_min = pedestal_left
                    Q_clip_max = pedestal_right
                
                    # Plot histograms with Q-specific clipping and bins
                    axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                        bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
                    axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min) & (y_B < Q_clip_max)], 
                                        bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
                    axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
                    axes_Q[i*4 + j].legend()
                    # Show between -5 and 5
                    axes_Q[i*4 + j].set_xlim([Q_clip_min, Q_clip_max])
            # Display a vertical green dashed, alpha = 0.5 line at 0
            for ax in axes_Q:
                ax.axvline(0, color='green', linestyle='--', alpha=0.5)
        
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.suptitle(f"Grand Figure for pedestal substracted values (zoom), mingo0{station}\n{start_time}", fontsize=16)
        
            if save_plots:
                final_filename = f'{fig_idx}_grand_figure_Q_pedestal_zoom_ST.png'
                fig_idx += 1
            
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
        
            if show_plots: plt.show()
            plt.close(fig_Q)
    
    
        if create_plots or create_essential_plots:
        # if create_plots:
            # ZOOOOOOOOOOOOOM ------------------------------------------------
            # Create the grand figure for Q values
            fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
            axes_Q = axes_Q.flatten()
        
            for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
                for j in range(4):
                    col_F = f'{key}_F_{j+1}'
                    col_B = f'{key}_B_{j+1}'
                    y_F = charge_test[col_F]
                    y_B = charge_test[col_B]
                
                    Q_clip_min = pedestal_left * 2
                    Q_clip_max = pedestal_right * 12
                
                    # Plot histograms with Q-specific clipping and bins
                    axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                        bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
                    axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min) & (y_B < Q_clip_max)], 
                                        bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
                    axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
                    axes_Q[i*4 + j].legend()
                    # Show between -5 and 5
                    axes_Q[i*4 + j].set_xlim([Q_clip_min, Q_clip_max])
            # Display a vertical green dashed, alpha = 0.5 line at 0
            for ax in axes_Q:
                ax.axvline(0, color='green', linestyle='--', alpha=0.5)
        
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.suptitle(f"Grand Figure for pedestal substracted values (zoom), mingo0{station}\n{start_time}", fontsize=16)
        
            if save_plots:
                final_filename = f'{fig_idx}_grand_figure_Q_pedestal_less_zoom_ST.png'
                fig_idx += 1
            
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
        
            if show_plots: plt.show()
            plt.close(fig_Q)


print("----------------------------------------------------------------------")
print("------------------- Position offset calibration ----------------------")
print("----------------------------------------------------------------------")

pos_test = working_df.copy()
for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
    for j in range(4):
        pos_test[f'{key}_diff_{j+1}'] = ( pos_test[f'{key}_B_{j+1}'] - pos_test[f'{key}_F_{j+1}'] ) / 2

pos_test_copy = pos_test.copy()
Tdiff_cal = []
for key in ['1', '2', '3', '4']:
    T_F_cols = [f'T{key}_F_{i+1}' for i in range(4)]
    T_F = working_df[T_F_cols].values
    
    T_B_cols = [f'T{key}_B_{i+1}' for i in range(4)]
    T_B = working_df[T_B_cols].values
    
    Tdiff_cal_component = [calibrate_strip_T_diff(T_F[:,i], T_B[:,i]) for i in range(4)]
    Tdiff_cal.append(Tdiff_cal_component)
Tdiff_cal = np.array(Tdiff_cal)

print("\nTime diff. offset:")
print(Tdiff_cal, "\n")

if validate_pos_cal:

    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            mask = pos_test_copy[f'{key}_diff_{j+1}'] != 0
            pos_test.loc[mask, f'{key}_diff_{j+1}'] -= Tdiff_cal[i][j]

    # if create_plots:
    if create_essential_plots or create_plots:
        # Create the grand figure for Q values
        fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_Q = axes_Q.flatten()
        
        for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
            for j in range(4):
                col_F = f'{key}_diff_{j+1}'
                y_F = pos_test[col_F]
                
                Q_clip_min = -2
                Q_clip_max = 2
                
                # Plot histograms with Q-specific clipping and bins
                axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                     bins=num_bins, alpha=0.5, label=f'{col_F}')
                axes_Q[i*4 + j].set_title(f'{col_F}')
                axes_Q[i*4 + j].legend()
                axes_Q[i*4 + j].set_xlabel('T_diff / ns')
                axes_Q[i*4 + j].set_xlim([Q_clip_min, Q_clip_max])
                
                # if log_scale:
                #     axes_Q[i*4 + j].set_yscale('log')  # For Q values
                
            for ax in axes_Q:
                ax.axvline(0, color='green', linestyle='--', alpha=0.5)
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Grand Figure for position calibration, new method, mingo0{station}\n{start_time}", fontsize=16)
        
        if save_plots:
            final_filename = f'{fig_idx}_grand_figure_T_diff_cal.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close(fig_Q)


if self_trigger:    
    print("----------------------------------------------------------------------")
    print("------------------- Position offset calibration ----------------------")
    print("----------------------------------------------------------------------")

    pos_test = working_st_df.copy()
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            pos_test[f'{key}_diff_{j+1}'] = ( pos_test[f'{key}_B_{j+1}'] - pos_test[f'{key}_F_{j+1}'] ) / 2

    pos_test_copy = pos_test.copy()
    Tdiff_cal_ST = []
    for key in ['1', '2', '3', '4']:
        T_F_cols = [f'T{key}_F_{i+1}' for i in range(4)]
        T_F = working_st_df[T_F_cols].values
    
        T_B_cols = [f'T{key}_B_{i+1}' for i in range(4)]
        T_B = working_st_df[T_B_cols].values
    
        Tdiff_cal_component = [calibrate_strip_T_diff(T_F[:,i], T_B[:,i], self_trigger_mode = self_trigger) for i in range(4)]
        Tdiff_cal_ST.append(Tdiff_cal_component)
    Tdiff_cal_ST = np.array(Tdiff_cal_ST)

    print("\nSELF TRIGGER Time diff. offset:")
    print(Tdiff_cal_ST, "\n")

    validate_pos_cal = True
    if validate_pos_cal:

        for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
            for j in range(4):
                mask = pos_test_copy[f'{key}_diff_{j+1}'] != 0
                pos_test.loc[mask, f'{key}_diff_{j+1}'] -= Tdiff_cal_ST[i][j]

        # if create_plots:
        if create_essential_plots or create_plots:
            # Create the grand figure for Q values
            fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
            axes_Q = axes_Q.flatten()
        
            for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
                for j in range(4):
                    col_F = f'{key}_diff_{j+1}'
                    y_F = pos_test[col_F]
                
                    Q_clip_min = -2
                    Q_clip_max = 2
                
                    # Plot histograms with Q-specific clipping and bins
                    axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                         bins=num_bins, alpha=0.5, label=f'{col_F}')
                    axes_Q[i*4 + j].set_title(f'{col_F}')
                    axes_Q[i*4 + j].legend()
                    axes_Q[i*4 + j].set_xlabel('T_diff / ns')
                    axes_Q[i*4 + j].set_xlim([Q_clip_min, Q_clip_max])
                
                    # if log_scale:
                    #     axes_Q[i*4 + j].set_yscale('log')  # For Q values
                
                for ax in axes_Q:
                    ax.axvline(0, color='green', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.suptitle(f"SELF TRIGGER Grand Figure for position calibration, new method, mingo0{station}\n{start_time}", fontsize=16)
        
            if save_plots:
                final_filename = f'{fig_idx}_grand_figure_T_diff_cal_ST.png'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
            if show_plots: plt.show()
            plt.close(fig_Q)


# ----------------------------------------------------------------------------------
# -------------------------- Semisums and semidifferences --------------------------
# ----------------------------------------------------------------------------------

for key in ['T1', 'T2', 'T3', 'T4']:
    T_F_cols = [f'{key}_F_{i+1}' for i in range(4)]
    T_B_cols = [f'{key}_B_{i+1}' for i in range(4)]
    Q_F_cols = [f'{key.replace("T", "Q")}_F_{i+1}' for i in range(4)]
    Q_B_cols = [f'{key.replace("T", "Q")}_B_{i+1}' for i in range(4)]

    T_F = working_df[T_F_cols].values
    T_B = working_df[T_B_cols].values
    Q_F = working_df[Q_F_cols].values
    Q_B = working_df[Q_B_cols].values

    new_cols = {}
    for i in range(4):
        # new_cols[f'{key}_T_sum_{i+1}'] = (T_F[:, i] + T_B[:, i]) / 2
        # new_cols[f'{key}_T_diff_{i+1}'] = (T_F[:, i] - T_B[:, i]) / 2
        # new_cols[f'{key.replace("T", "Q")}_Q_sum_{i+1}'] = (Q_F[:, i] + Q_B[:, i]) / 2
        # new_cols[f'{key.replace("T", "Q")}_Q_diff_{i+1}'] = (Q_F[:, i] - Q_B[:, i]) / 2
        
        new_cols[f'{key}_T_sum_{i+1}'] = (T_B[:, i] + T_F[:, i]) / 2
        new_cols[f'{key}_T_diff_{i+1}'] = (T_B[:, i] - T_F[:, i]) / 2
        new_cols[f'{key.replace("T", "Q")}_Q_sum_{i+1}'] = (Q_F[:, i] + Q_B[:, i]) / 2
        new_cols[f'{key.replace("T", "Q")}_Q_diff_{i+1}'] = (Q_F[:, i] - Q_B[:, i]) / 2

    working_df = pd.concat([working_df, pd.DataFrame(new_cols, index=working_df.index)], axis=1)


if self_trigger:

    for key in ['T1', 'T2', 'T3', 'T4']:
        T_F_cols = [f'{key}_F_{i+1}' for i in range(4)]
        T_B_cols = [f'{key}_B_{i+1}' for i in range(4)]
        Q_F_cols = [f'{key.replace("T", "Q")}_F_{i+1}' for i in range(4)]
        Q_B_cols = [f'{key.replace("T", "Q")}_B_{i+1}' for i in range(4)]

        T_F = working_st_df[T_F_cols].values
        T_B = working_st_df[T_B_cols].values
        Q_F = working_st_df[Q_F_cols].values
        Q_B = working_st_df[Q_B_cols].values

        new_cols = {}
        for i in range(4):
            new_cols[f'{key}_T_sum_{i+1}'] = (T_B[:, i] + T_F[:, i]) / 2
            new_cols[f'{key}_T_diff_{i+1}'] = (T_B[:, i] - T_F[:, i]) / 2
            new_cols[f'{key.replace("T", "Q")}_Q_sum_{i+1}'] = (Q_F[:, i] + Q_B[:, i]) / 2
            new_cols[f'{key.replace("T", "Q")}_Q_diff_{i+1}'] = (Q_F[:, i] - Q_B[:, i]) / 2

        working_st_df = pd.concat([working_st_df, pd.DataFrame(new_cols, index=working_st_df.index)], axis=1)

# if create_essential_plots or create_plots:
if create_plots:

    # Select only the columns that have 'Q_sum', 'Q_diff', 'T_sum', or 'T_diff' in their names
    plot_df = working_df.copy()
    plot_df = plot_df[[col for col in plot_df.columns if any(x in col for x in ['Q_sum', 'Q_diff', 'T_sum', 'T_diff'])]]
    
    num_columns = len(plot_df.columns) - 1  # Exclude 'datetime'
    num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
    fig, axes = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2))
    axes = axes.flatten()

    for i, col in enumerate([col for col in plot_df.columns if col != 'datetime']):
        y = plot_df[col]
        
        if 'Q_sum' in col:
            color = Q_sum_color
        elif 'Q_diff' in col:
            color = Q_diff_color
        elif 'T_sum' in col:
            color = T_sum_color
        elif 'T_diff' in col:
            color = T_diff_color
        else:
            print(col)
            continue
        axes[i].hist(y[y != 0], bins=100, alpha=0.5, label=col, color=color)
        axes[i].set_title(col)
        axes[i].legend()
        if 'Q_sum' in col:
            axes[i].set_yscale('log')
    
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at the top (5%)
    fig.suptitle("Uncalibrated data", fontsize=20)  # increase font size
    if save_plots:
        name_of_file = 'uncalibrated'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots: 
        plt.show()
    plt.close()


if time_window_filtering:
    
    print("----------------------------------------------------------------------")
    print("-------------------- Time window filtering (2/3) ---------------------")
    print("----------------------------------------------------------------------")
    
    # Pre removal of outliers
    spread_results = []
    for original_tt in sorted(working_df["original_tt"].unique()):
        filtered_df = working_df[working_df["original_tt"] == original_tt].copy()
        T_sum_columns_tt = filtered_df.filter(regex='_T_sum_').columns
        t_sum_spread_tt = filtered_df[T_sum_columns_tt].apply(lambda row: np.ptp(row[row != 0]) if np.any(row != 0) else np.nan, axis=1)
        filtered_df["T_sum_spread_OG"] = t_sum_spread_tt
        spread_results.append(filtered_df)
    spread_df = pd.concat(spread_results, ignore_index=True)

    # if create_plots:
    if create_essential_plots or create_plots:
        fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=False)
        axs = axs.flatten()
        for i, tt in enumerate(sorted(spread_df["original_tt"].unique())):
            subset = spread_df[spread_df["original_tt"] == tt]
            v = subset["T_sum_spread_OG"].dropna()
            v = v[v < coincidence_window_precal_ns * 3]
            axs[i].hist(v, bins=100, alpha=0.7)
            axs[i].set_title(f"TT = {tt}")
            axs[i].set_xlabel("ΔT (ns)")
            axs[i].set_ylabel("Events")
            axs[i].axvline(x=coincidence_window_precal_ns, color='red', linestyle='--', label='Time coincidence window')
            # Logscale
            axs[i].set_yscale('log')
        fig.suptitle("Non filtered. Intra-Event T_sum Spread by original_tt")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if save_plots:
            hist_filename = f'{fig_idx}_tsum_spread_histograms_OG.png'
            fig_idx += 1
            hist_path = os.path.join(base_directories["figure_directory"], hist_filename)
            plot_list.append(hist_path)
            fig.savefig(hist_path, format='png')
        if show_plots: plt.show()
        plt.close(fig)

    # Removal of outliers
    def zero_outlier_tsum(row, threshold=coincidence_window_precal_ns):
        t_sum_cols = [col for col in row.index if '_T_sum_' in col]
        t_sum_vals = row[t_sum_cols].copy()
        nonzero_vals = t_sum_vals[t_sum_vals != 0]
        if len(nonzero_vals) < 2: return row
        center = np.median(nonzero_vals)
        deviations = np.abs(nonzero_vals - center)
        outliers = deviations > threshold / 2
        for col in outliers.index[outliers]: row[col] = 0.0
        return row
    working_df = working_df.apply(zero_outlier_tsum, axis=1)

    # Post removal of outliers
    spread_results = []
    for original_tt in sorted(working_df["original_tt"].unique()):
        filtered_df = working_df[working_df["original_tt"] == original_tt].copy()
        T_sum_columns_tt = filtered_df.filter(regex='_T_sum_').columns
        t_sum_spread_tt = filtered_df[T_sum_columns_tt].apply(lambda row: np.ptp(row[row != 0]) if np.any(row != 0) else np.nan, axis=1)
        filtered_df["T_sum_spread_OG"] = t_sum_spread_tt
        spread_results.append(filtered_df)
    spread_df = pd.concat(spread_results, ignore_index=True)

    # if create_plots:
    if create_essential_plots or create_plots:
        fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=False)
        axs = axs.flatten()
        for i, tt in enumerate(sorted(spread_df["original_tt"].unique())):
            subset = spread_df[spread_df["original_tt"] == tt]
            v = subset["T_sum_spread_OG"].dropna()
            axs[i].hist(v, bins=100, alpha=0.7)
            axs[i].set_title(f"TT = {tt}")
            axs[i].set_xlabel("ΔT (ns)")
            axs[i].set_ylabel("Events")
            axs[i].axvline(x=coincidence_window_precal_ns, color='red', linestyle='--', label='Time coincidence window')# Logscale
            axs[i].set_yscale('log')
        fig.suptitle("Cleaned. Corrected Intra-Event T_sum Spread by original_tt")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if save_plots:
            hist_filename = f'{fig_idx}_tsum_spread_histograms_filtered_OG.png'
            fig_idx += 1
            hist_path = os.path.join(base_directories["figure_directory"], hist_filename)
            plot_list.append(hist_path)
            fig.savefig(hist_path, format='png')
        if show_plots: plt.show()
        plt.close(fig)


print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("--------------------- Filters and calibrations -----------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

print("-------------------- Filter 2: uncalibrated data ---------------------")

# FILTER 2: TSUM, TDIF, QSUM, QDIF PRECALIBRATED THRESHOLDS --> 0 if out ------------------------------
for col in working_df.columns:
    if 'T_sum' in col:
        working_df[col] = np.where((working_df[col] > T_sum_right_pre_cal) | (working_df[col] < T_sum_left_pre_cal), 0, working_df[col])
    if 'T_diff' in col:
        working_df[col] = np.where((working_df[col] > T_diff_pre_cal_threshold) | (working_df[col] < -T_diff_pre_cal_threshold), 0, working_df[col])
    if 'Q_sum' in col:
        working_df[col] = np.where((working_df[col] > Q_right_pre_cal) | (working_df[col] < Q_left_pre_cal), 0, working_df[col])
    if 'Q_diff' in col:
        working_df[col] = np.where((working_df[col] > Q_diff_pre_cal_threshold) | (working_df[col] < -Q_diff_pre_cal_threshold), 0, working_df[col])


# if create_essential_plots or create_plots:
if create_plots:

    # Select only the columns that have 'Q_sum', 'Q_diff', 'T_sum', or 'T_diff' in their names
    plot_df = working_df.copy()
    plot_df = plot_df[[col for col in plot_df.columns if any(x in col for x in ['Q_sum', 'Q_diff', 'T_sum', 'T_diff'])]]
    
    num_columns = len(plot_df.columns) - 1  # Exclude 'datetime'
    num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
    fig, axes = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2))
    axes = axes.flatten()

    for i, col in enumerate([col for col in plot_df.columns if col != 'datetime']):
        y = plot_df[col]
        
        if 'Q_sum' in col:
            color = Q_sum_color
        elif 'Q_diff' in col:
            color = Q_diff_color
        elif 'T_sum' in col:
            color = T_sum_color
        elif 'T_diff' in col:
            color = T_diff_color
        else:
            print(col)
            continue
        axes[i].hist(y[y != 0], bins=100, alpha=0.5, label=col, color=color)
        axes[i].set_title(col)
        axes[i].legend()
        if 'Q_sum' in col:
            axes[i].set_yscale('log')
    
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at the top (5%)
    fig.suptitle("Uncalibrated data, filtered", fontsize=20)  # increase font size
    if save_plots:
        name_of_file = 'uncalibrated_filtered'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: 
        plt.show()
    plt.close()


print("----------------------------------------------------------------------")
print("----------- Charge sum pedestal, calibration and filtering -----------")
print("----------------------------------------------------------------------")

for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    for j in range(4):
        mask = working_df[f'{key}_Q_sum_{j+1}'] != 0
        # working_df.loc[mask, f'{key}_Q_sum_{j+1}'] -= calibration_Q[i][j]
        working_df.loc[mask, f'{key}_Q_sum_{j+1}'] -= ( QF_pedestal[i][j] + QB_pedestal[i][j] ) / 2

print("------------------ Filter 3: charge sum filtering --------------------")
for col in working_df.columns:
    if 'Q_sum' in col:
        working_df[col] = np.where((working_df[col] > Q_sum_right_cal) | (working_df[col] < Q_sum_left_cal), 0, working_df[col])


if self_trigger: 
    for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        for j in range(4):
            mask = working_st_df[f'{key}_Q_sum_{j+1}'] != 0
            working_st_df.loc[mask, f'{key}_Q_sum_{j+1}'] -= ( QF_pedestal_ST[i][j] + QB_pedestal_ST[i][j] ) / 2
    for col in working_st_df.columns:
        if 'Q_sum' in col:
            working_st_df[col] = np.where((working_st_df[col] > Q_sum_right_cal) | (working_st_df[col] < Q_sum_left_cal), 0, working_st_df[col])


print("----------------------------------------------------------------------")
print("----------------- Time diff calibration and filtering ----------------")
print("----------------------------------------------------------------------")

for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
    for j in range(4):
        mask = working_df[f'{key}_T_diff_{j+1}'] != 0
        working_df.loc[mask, f'{key}_T_diff_{j+1}'] -= Tdiff_cal[i][j]

print("--------------------- Filter 3.2: time diff filtering ----------------")
for col in working_df.columns:
    if 'T_diff' in col:
        working_df[col] = np.where((working_df[col] > T_diff_cal_threshold) | (working_df[col] < -T_diff_cal_threshold), 0, working_df[col])


if self_trigger:
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            mask = working_st_df[f'{key}_T_diff_{j+1}'] != 0
            working_st_df.loc[mask, f'{key}_T_diff_{j+1}'] -= Tdiff_cal_ST[i][j]
    for col in working_st_df.columns:
        if 'T_diff' in col:
            working_st_df[col] = np.where((working_st_df[col] > T_diff_cal_threshold) | (working_st_df[col] < -T_diff_cal_threshold), 0, working_st_df[col])


print("----------------------------------------------------------------------")
print("---------------- Charge diff calibration and filtering ---------------")
print("----------------------------------------------------------------------")

for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    for j in range(4):
        mask = working_df[f'{key}_Q_diff_{j+1}'] != 0
        # working_df.loc[mask, f'{key}_Q_diff_{j+1}'] -= calibration_Q_FB[i][j]
        working_df.loc[mask, f'{key}_Q_diff_{j+1}'] -= ( QF_pedestal[i][j] - QB_pedestal[i][j] ) / 2

print("------------------ Filter 4: charge diff filtering -------------------")
for col in working_df.columns:
    if 'Q_diff' in col:
        working_df[col] = np.where((working_df[col] > Q_diff_cal_threshold) | (working_df[col] < -Q_diff_cal_threshold), 0, working_df[col])


if self_trigger:
    
    for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        for j in range(4):
            mask = working_st_df[f'{key}_Q_diff_{j+1}'] != 0
            working_st_df.loc[mask, f'{key}_Q_diff_{j+1}'] -= ( QF_pedestal_ST[i][j] - QB_pedestal_ST[i][j] ) / 2

    print("------------------ Filter 4: charge diff filtering -------------------")
    for col in working_st_df.columns:
        if 'Q_diff' in col:
            working_st_df[col] = np.where((working_st_df[col] > Q_diff_cal_threshold) | (working_st_df[col] < -Q_diff_cal_threshold), 0, working_st_df[col])


# if create_essential_plots or create_plots:
if create_plots:

    # Select only the columns that have 'Q_sum', 'Q_diff', 'T_sum', or 'T_diff' in their names
    plot_df = working_df.copy()
    plot_df = plot_df[[col for col in plot_df.columns if any(x in col for x in ['Q_sum', 'Q_diff', 'T_sum', 'T_diff'])]]
    
    num_columns = len(plot_df.columns) - 1  # Exclude 'datetime'
    num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
    fig, axes = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2))
    axes = axes.flatten()

    for i, col in enumerate([col for col in plot_df.columns if col != 'datetime']):
        y = plot_df[col]
        
        if 'Q_sum' in col:
            color = Q_sum_color
        elif 'Q_diff' in col:
            color = Q_diff_color
        elif 'T_sum' in col:
            color = T_sum_color
        elif 'T_diff' in col:
            color = T_diff_color
        else:
            print(col)
            continue
        axes[i].hist(y[y != 0], bins=100, alpha=0.5, label=col, color=color)
        axes[i].set_title(col)
        axes[i].legend()
        if 'Q_sum' in col:
            axes[i].set_yscale('log')
    
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at the top (5%)
    fig.suptitle("Calibrated filtered data before FB correction", fontsize=20)  # increase font size
    if save_plots:
        name_of_file = 'calibrated_filtered_before_FB_corr'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: 
        plt.show()
    plt.close()


print("----------------------------------------------------------------------")
print("------------------- Charge front-back correction ---------------------")
print("----------------------------------------------------------------------")

if charge_front_back:
    for key in [1, 2, 3, 4]:
        for i in range(4):
            # Extract data from the DataFrame
            Q_sum = working_df[f'Q{key}_Q_sum_{i+1}'].values
            Q_diff = working_df[f'Q{key}_Q_diff_{i+1}'].values

            # Apply condition to filter non-zero Q_sum and Q_diff
            cond = (Q_sum != 0) & (Q_diff != 0)
            Q_sum_adjusted = Q_sum[cond]
            Q_diff_adjusted = Q_diff[cond]
            
            # Skip correction if no data is left after filtering
            if np.sum(Q_sum_adjusted) == 0:
                continue

            # Perform scatter plot and fit
            title = f"Q{key}_{i+1}. Charge diff. vs. charge sum."
            x_label = "Charge sum"
            y_label = "Charge diff"
            name_of_file = f"Q{key}_{i+1}_charge_analysis_scatter_diff_vs_sum"
            coeffs = scatter_2d_and_fit_new(Q_sum_adjusted, Q_diff_adjusted, title, x_label, y_label, name_of_file)
            print([f"{coeff:.3g}" for coeff in coeffs])
            working_df.loc[cond, f'Q{key}_Q_diff_{i+1}'] = Q_diff_adjusted - polynomial(Q_sum_adjusted, *coeffs)
    
    if self_trigger:
        print("SELF TRIGGER Charge front-back correction...")
        for key in [1, 2, 3, 4]:
            for i in range(4):
                # Extract data from the DataFrame
                Q_sum = working_st_df[f'Q{key}_Q_sum_{i+1}'].values
                Q_diff = working_st_df[f'Q{key}_Q_diff_{i+1}'].values

                # Apply condition to filter non-zero Q_sum and Q_diff
                cond = (Q_sum != 0) & (Q_diff != 0)
                Q_sum_adjusted = Q_sum[cond]
                Q_diff_adjusted = Q_diff[cond]
            
                # Skip correction if no data is left after filtering
                if np.sum(Q_sum_adjusted) == 0:
                    continue

                # Perform scatter plot and fit
                title = f"Q{key}_{i+1}. SELF TRIGGER Charge diff. vs. charge sum."
                x_label = "Charge sum"
                y_label = "Charge diff"
                name_of_file = f"Q{key}_{i+1}_charge_analysis_scatter_diff_vs_sum_ST"
                coeffs = scatter_2d_and_fit_new(Q_sum_adjusted, Q_diff_adjusted, title, x_label, y_label, name_of_file)
                working_st_df.loc[cond, f'Q{key}_Q_diff_{i+1}'] = Q_diff_adjusted - polynomial(Q_sum_adjusted, *coeffs)
        
    print('\nCharge front-back correction performed.')
    
else:
    print('Charge front-back correction was selected to not be performed.')
    Q_diff_cal_threshold_FB = Q_diff_cal_threshold_FB_wide


print("----------------------------------------------------------------------")
print("------------- Filter 5: charge difference FB filtering ---------------")
for col in working_df.columns:
    if 'Q_diff' in col:
        working_df[col] = np.where(np.abs(working_df[col]) < Q_diff_cal_threshold_FB, working_df[col], 0)


if self_trigger:
    for col in working_st_df.columns:
        if 'Q_diff' in col:
            working_st_df[col] = np.where(np.abs(working_st_df[col]) < Q_diff_cal_threshold_FB, working_st_df[col], 0)


# if create_essential_plots or create_plots:
if create_plots:

    # Select only the columns that have 'Q_sum', 'Q_diff', 'T_sum', or 'T_diff' in their names
    plot_df = working_df.copy()
    plot_df = plot_df[[col for col in plot_df.columns if any(x in col for x in ['Q_sum', 'Q_diff', 'T_sum', 'T_diff'])]]
    
    num_columns = len(plot_df.columns) - 1  # Exclude 'datetime'
    num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
    fig, axes = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2))
    axes = axes.flatten()

    for i, col in enumerate([col for col in plot_df.columns if col != 'datetime']):
        y = plot_df[col]
        
        if 'Q_sum' in col:
            color = Q_sum_color
        elif 'Q_diff' in col:
            color = Q_diff_color
        elif 'T_sum' in col:
            color = T_sum_color
        elif 'T_diff' in col:
            color = T_diff_color
        else:
            print(col)
            continue
        axes[i].hist(y[y != 0], bins=100, alpha=0.5, label=col, color=color)
        axes[i].set_title(col)
        # axes[i].legend()
        if 'Q_sum' in col:
            axes[i].set_yscale('log')
    
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at the top (5%)
    fig.suptitle("Calibrated filtered data including FB correction", fontsize=20)  # increase font size
    if save_plots:
        name_of_file = 'calibrated_filtered'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: 
        plt.show()
    plt.close()


print("----------------------------------------------------------------------")
print("---------- Filter if any variable in the strip is 0 (1/3) ------------")
print("----------------------------------------------------------------------")

# Now go throuhg every plane and strip and if any of the T_sum, T_diff, Q_sum, Q_diff == 0,
# put the four variables in that plane, strip and event to 0

total_events = len(working_df)

for plane in range(1, 5):
    for strip in range(1, 5):
        q_sum  = f'Q{plane}_Q_sum_{strip}'
        q_diff = f'Q{plane}_Q_diff_{strip}'
        t_sum  = f'T{plane}_T_sum_{strip}'
        t_diff = f'T{plane}_T_diff_{strip}'
        
        # Build mask
        mask = (
            (working_df[q_sum]  == 0) |
            (working_df[q_diff] == 0) |
            (working_df[t_sum]  == 0) |
            (working_df[t_diff] == 0)
        )
        
        # Count affected events
        num_affected_events = mask.sum()
        print(f"Plane {plane}, Strip {strip}: {num_affected_events} out of {total_events} events affected ({(num_affected_events / total_events) * 100:.2f}%)")

        # Zero the affected values
        working_df.loc[mask, [q_sum, q_diff, t_sum, t_diff]] = 0


if self_trigger:
    
    total_events = len(working_st_df)

    for plane in range(1, 5):
        for strip in range(1, 5):
            q_sum  = f'Q{plane}_Q_sum_{strip}'
            q_diff = f'Q{plane}_Q_diff_{strip}'
            t_sum  = f'T{plane}_T_sum_{strip}'
            t_diff = f'T{plane}_T_diff_{strip}'
        
            # Build mask
            mask = (
                (working_st_df[q_sum]  == 0) |
                (working_st_df[q_diff] == 0) |
                (working_st_df[t_sum]  == 0) |
                (working_st_df[t_diff] == 0)
            )
        
            # Count affected events
            num_affected_events = mask.sum()
            print(f"SELF TRIGGER. Plane {plane}, Strip {strip}: {num_affected_events} out of {total_events} events affected ({(num_affected_events / total_events) * 100:.2f}%)")

            # Zero the affected values
            working_st_df.loc[mask, [q_sum, q_diff, t_sum, t_diff]] = 0


if create_essential_plots or create_plots:
# if create_plots:

    # Select only the columns that have 'Q_sum', 'Q_diff', 'T_sum', or 'T_diff' in their names
    plot_df = working_df.copy()
    plot_df = plot_df[[col for col in plot_df.columns if any(x in col for x in ['Q_sum', 'Q_diff', 'T_sum', 'T_diff'])]]
    
    num_columns = len(plot_df.columns) - 1  # Exclude 'datetime'
    num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
    fig, axes = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2))
    axes = axes.flatten()

    for i, col in enumerate([col for col in plot_df.columns if col != 'datetime']):
        y = plot_df[col]
        
        if 'Q_sum' in col:
            color = Q_sum_color
        elif 'Q_diff' in col:
            color = Q_diff_color
        elif 'T_sum' in col:
            color = T_sum_color
        elif 'T_diff' in col:
            color = T_diff_color
        else:
            print(col)
            continue
        axes[i].hist(y[y != 0], bins=100, alpha=0.5, label=col, color=color)
        axes[i].set_title(col)
        # axes[i].legend()
        if 'Q_sum' in col:
            axes[i].set_yscale('log')
    
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at the top (5%)
    fig.suptitle("Calibrated filtered data including FB correction removing zeroes in any variable", fontsize=20)  # increase font size
    if save_plots:
        name_of_file = 'calibrated_filtered_removed_zeroes'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: 
        plt.show()
    plt.close()


if self_trigger:
    if create_essential_plots or create_plots:
    # if create_plots:

        # Select only the columns that have 'Q_sum', 'Q_diff', 'T_sum', or 'T_diff' in their names
        plot_df = working_st_df.copy()
        plot_df = plot_df[[col for col in plot_df.columns if any(x in col for x in ['Q_sum', 'Q_diff', 'T_sum', 'T_diff'])]]
    
        num_columns = len(plot_df.columns) - 1  # Exclude 'datetime'
        num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
        fig, axes = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2))
        axes = axes.flatten()

        for i, col in enumerate([col for col in plot_df.columns if col != 'datetime']):
            y = plot_df[col]
        
            if 'Q_sum' in col:
                color = Q_sum_color
            elif 'Q_diff' in col:
                color = Q_diff_color
            elif 'T_sum' in col:
                color = T_sum_color
            elif 'T_diff' in col:
                color = T_diff_color
            else:
                print(col)
                continue
            axes[i].hist(y[y != 0], bins=100, alpha=0.5, label=col, color=color)
            axes[i].set_title(col)
            # axes[i].legend()
            if 'Q_sum' in col:
                axes[i].set_yscale('log')
    
        # Remove any unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
    
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at the top (5%)
        fig.suptitle("SELF TRIGGER Calibrated filtered data including FB correction removing zeroes in any variable", fontsize=20)  # increase font size
        if save_plots:
            name_of_file = 'calibrated_filtered_removed_zeroes_ST'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1

            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
    
        if show_plots: 
            plt.show()
        plt.close()




if slewing_correction:
    
    print("----------------------------------------------------------------------")
    print("---------------------- Slewing correction 1/2 ------------------------")
    print("----------------------------------------------------------------------")
    
    # Select desired columns
    cols = working_df.columns
    t_sum_cols   = [c for c in cols if 'T_sum' in c and '_final' not in c]
    q_sum_cols   = [c for c in cols if 'Q_sum' in c and '_final' not in c]
    t_diff_cols  = [c for c in cols if 'T_diff' in c and '_final' not in c]
    type_col     = ['type'] if 'type' in cols else []

    data_df_times   = working_df[t_sum_cols]
    data_df_charges = working_df[q_sum_cols]
    data_df_tdiff   = working_df[t_diff_cols]
    type_series     = working_df[type_col] if type_col else None

    # Concatenate all relevant data with 'type' column
    data_df_filt = pd.concat([data_df_charges, data_df_times, data_df_tdiff], axis=1)
    data_slew = data_df_filt
    
    # Select y_pos for each plane
    y_lookup = {
        1: y_pos_T[0],
        2: y_pos_T[1],
        3: y_pos_T[0],
        4: y_pos_T[1],
    }
    
    results = []
    
    # Loop through all combinations of planes and strips
    for (p1, s1), (p2, s2) in combinations([(p, s) for p in range(1, 5) for s in range(1, 5)], 2):
        Q1 = data_slew[f"Q{p1}_Q_sum_{s1}"]
        Q2 = data_slew[f"Q{p2}_Q_sum_{s2}"]
        T1 = data_slew[f"T{p1}_T_sum_{s1}"]
        T2 = data_slew[f"T{p2}_T_sum_{s2}"]
        TD1 = data_slew[f"T{p1}_T_diff_{s1}"]
        TD2 = data_slew[f"T{p2}_T_diff_{s2}"]
        
        valid_mask = (
            (Q1 != 0) & (Q2 != 0) &
            (T1 != 0) & (T2 != 0) &
            (TD1 != 0) & (TD2 != 0)
        )

        # Apply mask to compute only valid values
        Q1 = Q1[valid_mask]
        Q2 = Q2[valid_mask]
        T1 = T1[valid_mask]
        T2 = T2[valid_mask]
        TD1 = TD1[valid_mask]
        TD2 = TD2[valid_mask]
        
        x1 = TD1 * tdiff_to_x  # mm
        x2 = TD2 * tdiff_to_x
        y1 = y_lookup[p1][s1 - 1]
        y2 = y_lookup[p2][s2 - 1]
        z1 = z_positions[p1 - 1]
        z2 = z_positions[p2 - 1]

        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2

        travel_time = np.sqrt(dx**2 + dy**2 + dz**2) / c_mm_ns
        tsum_diff = (T1 - T2)
        corrected_tsum_diff = tsum_diff + travel_time

        results.append(pd.DataFrame({
            'plane1': p1, 'strip1': s1,
            'plane2': p2, 'strip2': s2,
            'Q_sum_semidiff': 0.5 * (Q1 - Q2),
            'Q_sum_semisum':  0.5 * (Q1 + Q2),
            'T_sum_corrected_diff': corrected_tsum_diff,
            'T_sum_diff': tsum_diff,
            'x_diff': dx,
            'travel_time': travel_time
        }))

    # Concatenate all results
    slew_df = pd.concat(results, ignore_index=True)
    
    
    # -------------------------------------------------------------------
    # dx vs Time Differences
    # -------------------------------------------------------------------
    
    # if create_essential_plots or create_plots:
    if create_plots:

        pair_labels = [
            (p1, s1, p2, s2)
            for p1 in range(1, 5)
            for s1 in range(1, 5)
            for p2 in range(p1 + 1, 5)
            for s2 in range(1, 5)
        ]

        # Parameters
        batch_size = 4  # number of pairs per batch
        num_batches = int(np.ceil(len(pair_labels) / batch_size))

        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(pair_labels))
            current_pairs = pair_labels[start_idx:end_idx]

            fig, axes = plt.subplots(1, batch_size, figsize=(6 * batch_size, 5), constrained_layout=True)
            axes = np.atleast_2d(axes)

            for col_idx, (p1, s1, p2, s2) in enumerate(current_pairs):
                mask = (
                    (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
                    (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
                )
                data = slew_df[mask]

                # Only use rows with valid x_diff and T values
                valid_mask = (
                    (data['x_diff'] != 0) &
                    (data['T_sum_corrected_diff'] != 0) &
                    (data['T_sum_diff'] != 0)
                )
                data = data[valid_mask]

                x = data['x_diff']
                t_uncorrected = data['T_sum_diff']
                t_corrected = data['T_sum_corrected_diff']

                # dx vs T_sum_diff (uncorrected)
                ax1 = axes[0, col_idx]
                ax1.scatter(x, t_uncorrected, s=5, alpha=0.6, color='tab:red')
                ax1.set_title(f"P{p1}S{s1} vs P{p2}S{s2} — Uncorrected")
                ax1.set_xlabel("dx (mm)")
                ax1.set_ylabel("T_sum_diff (ns)")
                ax1.set_xlim([-300, 300])
                ax1.set_ylim([-5, 5])

                ax1.scatter(x, t_corrected, s=5, alpha=0.6, color='tab:blue')
                ax1.set_title(f"P{p1}S{s1} vs P{p2}S{s2} — Corrected")
                ax1.grid(True, linestyle='--', alpha=0.5)

            # Hide unused subplots
            for row in range(1):
                for col in range(len(current_pairs), batch_size):
                    axes[row, col].set_visible(False)

            fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at the top (5%)
            fig.suptitle(f"Batch {batch + 1}/{num_batches} — dx vs Time Differences", fontsize=16)  # increase font size

            if save_plots:
                name_of_file = 'dx_vs_tsum'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1

                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')

            if show_plots:
                plt.show()
            plt.close()

    
    # -------------------------------------------------------------------
    # dx vs Travel Time
    # -------------------------------------------------------------------
    
    # if create_essential_plots or create_plots:
    if create_plots:
        pair_labels = [
            (p1, s1, p2, s2)
            for p1 in range(1, 5)
            for s1 in range(1, 5)
            for p2 in range(p1 + 1, 5)
            for s2 in range(1, 5)
        ]

        # Parameters
        batch_size = 4  # number of pairs per batch
        num_batches = int(np.ceil(len(pair_labels) / batch_size))
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(pair_labels))
            current_pairs = pair_labels[start_idx:end_idx]

            fig, axes = plt.subplots(1, batch_size, figsize=(6 * batch_size, 5), constrained_layout=True)
            axes = np.atleast_1d(axes)

            for col_idx, (p1, s1, p2, s2) in enumerate(current_pairs):
                mask = (
                    (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
                    (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
                )
                data = slew_df[mask]

                # Valid entries only
                valid_mask = (
                    (data['x_diff'] != 0) &
                    (data['travel_time'] != 0)
                )
                data = data[valid_mask]

                x = data['x_diff']
                t = data['travel_time']

                ax = axes[col_idx]
                ax.scatter(x, t, s=5, alpha=0.6, color='tab:green')
                ax.set_title(f"P{p1}S{s1} vs P{p2}S{s2}")
                ax.set_xlabel("dx (mm)")
                ax.set_ylabel("travel_time (ns)")
                ax.set_xlim([-300, 300])
                ax.set_ylim([0.4, 1.85])
                ax.grid(True, linestyle='--', alpha=0.5)

            # Hide unused subplots
            for col in range(len(current_pairs), batch_size):
                axes[col].set_visible(False)
            
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at the top (5%)
            fig.suptitle(f"Batch {batch + 1}/{num_batches} — dx vs Travel Time", fontsize=16)  # increase font size
            
            if save_plots:
                name_of_file = 'dx_vs_travel_time'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1

                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')

            if show_plots:
                plt.show()
            plt.close()


    # -------------------------------------------------------------------
    # Slewing histograms
    # -------------------------------------------------------------------
    
    # if create_essential_plots or create_plots:
    if create_plots:
        
        pair_labels = [
            (p1, s1, p2, s2)
            for p1 in range(1, 5)
            for s1 in range(1, 5)
            for p2 in range(p1 + 1, 5)
            for s2 in range(1, 5)
        ]

        # Parameters
        batch_size = 4  # number of pairs per batch
        num_batches = int(np.ceil(len(pair_labels) / batch_size))

        # ---- Loop through batches of pairs and plot all three histograms per pair ----
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(pair_labels))
            current_pairs = pair_labels[start_idx:end_idx]

            fig, axes = plt.subplots(3, batch_size, figsize=(6 * batch_size, 12), constrained_layout=True)
            axes = np.atleast_2d(axes)

            for col_idx, (p1, s1, p2, s2) in enumerate(current_pairs):
                mask = (
                    (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
                    (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
                )
                data = slew_df[mask]

                # Plot Q_sum_semidiff
                sns.histplot(data['Q_sum_semidiff'], bins=100, kde=True, ax=axes[0, col_idx], element='bars', edgecolor=None)
                axes[0, col_idx].set_title(f"P{p1}S{s1} vs P{p2}S{s2} — Q_diff")
                axes[0, col_idx].set_xlabel("Q_sum_semidiff")
                axes[0, col_idx].set_ylabel("Counts")
                axes[0, col_idx].set_xlim([-40, 40])

                # Plot Q_sum_semisum
                sns.histplot(data['Q_sum_semisum'], bins=100, kde=True, ax=axes[1, col_idx], element='bars', edgecolor=None)
                axes[1, col_idx].set_title(f"P{p1}S{s1} vs P{p2}S{s2} — Q_sum")
                axes[1, col_idx].set_xlabel("Q_sum_semisum")
                axes[1, col_idx].set_ylabel("Counts")
                axes[1, col_idx].set_xlim([0, 60])

                # Plot T_sum_corrected_diff
                sns.histplot(data['T_sum_corrected_diff'], bins=100, kde=True, ax=axes[2, col_idx], element='bars', edgecolor=None)
                axes[2, col_idx].set_title(f"P{p1}S{s1} vs P{p2}S{s2} — ΔT corrected")
                axes[2, col_idx].set_xlabel("T_sum_corrected_diff")
                axes[2, col_idx].set_ylabel("Counts")
                axes[2, col_idx].set_xlim([-5, 5])

            # Hide unused subplots
            for row in range(3):
                for col in range(len(current_pairs), batch_size):
                    axes[row, col].set_visible(False)

            fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at the top (5%)
            fig.suptitle(f"Batch {batch + 1}/{num_batches} — Histograms per Plane/Strip Pair", fontsize=16)  # increase font size
            
            if save_plots:
                name_of_file = 'slewing'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1

                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
        
            if show_plots: 
                plt.show()
            plt.close()
            
    
    # -------------------------------------------------------------------
    # 3D Slewing Observables
    # -------------------------------------------------------------------
    
    # if create_essential_plots or create_plots:
    if create_plots:
        
        pair_labels = [
            (p1, s1, p2, s2)
            for p1 in range(1, 5)
            for s1 in range(1, 5)
            for p2 in range(p1 + 1, 5)
            for s2 in range(1, 5)
        ]

        # Parameters
        batch_size = 4  # number of pairs per batch
        num_batches = int(np.ceil(len(pair_labels) / batch_size))
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(pair_labels))
            current_pairs = pair_labels[start_idx:end_idx]

            fig = plt.figure(figsize=(6 * batch_size, 16), constrained_layout=True)
            spec = gridspec.GridSpec(nrows=4, ncols=batch_size, figure=fig)

            for col_idx, (p1, s1, p2, s2) in enumerate(current_pairs):
                mask = (
                    (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
                    (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
                )
                data = slew_df[mask]

                # Drop rows with any invalid values for this pair
                valid_mask = (
                    (data['Q_sum_semidiff'] != 0) &
                    (data['Q_sum_semisum'] != 0) &
                    (data['T_sum_corrected_diff'] != 0)
                )
                data = data[valid_mask]

                x = data['T_sum_corrected_diff']
                y = data['Q_sum_semisum']
                z = data['Q_sum_semidiff']

                # 3D plot
                ax3d = fig.add_subplot(spec[0:2, col_idx], projection='3d')
                ax3d.scatter(x, y, z, s=5, alpha=0.6)
                ax3d.set_title(f"P{p1}S{s1} vs P{p2}S{s2}")
                ax3d.set_xlabel('ΔT corrected (ns)')
                ax3d.set_ylabel('Q_sum_semisum')
                ax3d.set_zlabel('Q_sum_semidiff')
                

                ax3d.set_xlim([delta_t_left, delta_t_right])
                ax3d.set_ylim([q_sum_left, q_sum_right])
                ax3d.set_zlim([q_diff_left, q_diff_right])

                # XY projection
                ax_xy = fig.add_subplot(spec[2, col_idx])
                ax_xy.scatter(x, y, s=5, alpha=0.5)
                ax_xy.set_xlabel('ΔT corrected')
                ax_xy.set_ylabel('Q_sum_semisum')
                ax_xy.set_title('XY projection')
                ax_xy.set_xlim([delta_t_left, delta_t_right])
                ax_xy.set_ylim([q_sum_left, q_sum_right])

                # XZ projection
                ax_xz = fig.add_subplot(spec[3, col_idx])
                ax_xz.scatter(x, z, s=5, alpha=0.5, c='tab:red')
                ax_xz.set_xlabel('ΔT corrected')
                ax_xz.set_ylabel('Q_sum_semidiff')
                ax_xz.set_title('XZ projection')
                ax_xz.set_xlim([delta_t_left, delta_t_right])
                ax_xz.set_ylim([q_diff_left, q_diff_right])

            fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at the top (5%)
            fig.suptitle(f"Batch {batch + 1}/{num_batches} — 3D Slewing Observables", fontsize=16)  # increase font size

            if save_plots:
                name_of_file = 'slewing_3d'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1

                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')

            if show_plots:
                plt.show()
            plt.close()


    # THE FIT ----------------------------------------------------------------
    
    pair_labels = [
            (p1, s1, p2, s2)
            for p1 in range(1, 5)
            for s1 in range(1, 5)
            for p2 in range(p1 + 1, 5)
            for s2 in range(1, 5)
        ]
    
    # Store fitted model parameters
    fit_results = []

    def robust_z_filter(df, cols, threshold=3.5):

        mask = np.ones(len(df), dtype=bool)
        for col in cols:
            median = np.median(df[col])
            mad = median_abs_deviation(df[col], scale='normal')  # consistent with std if normal
            if mad == 0:
                continue  # skip flat distributions
            z_mod = 0.6745 * (df[col] - median) / mad
            mask &= np.abs(z_mod) < threshold
        return df[mask]
    
    for (p1, s1, p2, s2) in pair_labels:
        mask = (
            (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
            (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
        )
        data = slew_df[mask]

        valid_mask = (
            (data['Q_sum_semidiff'] != 0) &
            (data['Q_sum_semisum'] != 0) &
            (data['T_sum_corrected_diff'] != 0)
        )
        data = data[valid_mask]

        if len(data) < 10:
            continue  # not enough data to fit

        # Apply some filtering on the values of Q_sum_semidiff and Q_sum_semisum
        data = data[
            (data['Q_sum_semidiff'] > Q_sum_semidiff_left) & (data['Q_sum_semidiff'] < Q_sum_semidiff_right) &
            (data['Q_sum_semisum'] > Q_sum_semisum_left) & (data['Q_sum_semisum'] < Q_sum_semisum_right) &
            (data['T_sum_corrected_diff'] > T_sum_corrected_diff_left) & (data['T_sum_corrected_diff'] < T_sum_corrected_diff_right)
        ]
        
        # Apply it to your DataFrame:
        # data = robust_z_filter(data, ['Q_sum_semidiff', 'Q_sum_semisum', 'T_sum_corrected_diff'])
        
        if not_use_q_semisum:
            X = data[['Q_sum_semidiff']].values
            y = data['T_sum_corrected_diff'].values

            model = LinearRegression()
            model.fit(X, y)
            
            b_semidiff = model.coef_[0]
            a_semisum = 0
            r2_score_val = model.score(X, y)
        else:
            X = data[['Q_sum_semisum', 'Q_sum_semidiff']].values
            y = data['T_sum_corrected_diff'].values

            model = LinearRegression()
            model.fit(X, y)
            
            a_semisum = model.coef_[0]
            b_semidiff = model.coef_[1]
            r2_score_val = model.score(X, y)

        # Store results
        fit_results.append({
            'plane1': p1, 'strip1': s1,
            'plane2': p2, 'strip2': s2,
            'a_semisum': a_semisum,
            'b_semidiff': b_semidiff,
            'c_offset': model.intercept_,
            'n_points': len(data),
            'r2_score': r2_score_val
        })

    # Create dataframe with all model parameters
    slewing_fit_df = pd.DataFrame(fit_results)
    
    print("Fitting results:")
    print(slewing_fit_df)
    
    
    # -------------------------------------------------------------------
    # 3D Slewing with fit projections
    # -------------------------------------------------------------------
    
    # if create_essential_plots or create_plots:
    if create_plots:

        pair_labels = [
            (p1, s1, p2, s2)
            for p1 in range(1, 5)
            for s1 in range(1, 5)
            for p2 in range(p1 + 1, 5)
            for s2 in range(1, 5)
        ]

        batch_size = 8
        num_batches = int(np.ceil(len(pair_labels) / batch_size))

        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(pair_labels))
            current_pairs = pair_labels[start_idx:end_idx]

            fig = plt.figure(figsize=(6 * batch_size, 16), constrained_layout=True)
            spec = gridspec.GridSpec(nrows=4, ncols=batch_size, figure=fig)

            for col_idx, (p1, s1, p2, s2) in enumerate(current_pairs):
                mask = (
                    (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
                    (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
                )
                data = slew_df[mask]

                valid_mask = (
                    (data['Q_sum_semidiff'] != 0) &
                    (data['Q_sum_semisum'] != 0) &
                    (data['T_sum_corrected_diff'] != 0)
                )
                data = data[valid_mask]

                if len(data) < 10:
                    continue

                # Retrieve fitted model coefficients from slewing_fit_df
                fit_row = slewing_fit_df[
                    (slewing_fit_df['plane1'] == p1) &
                    (slewing_fit_df['strip1'] == s1) &
                    (slewing_fit_df['plane2'] == p2) &
                    (slewing_fit_df['strip2'] == s2)
                ]
                if fit_row.empty:
                    continue

                a = fit_row['a_semisum'].values[0]
                b = fit_row['b_semidiff'].values[0]
                c = fit_row['c_offset'].values[0]

                x = data['T_sum_corrected_diff'].values
                y = data['Q_sum_semisum'].values
                z = data['Q_sum_semidiff'].values

                # 3D plot
                ax3d = fig.add_subplot(spec[0:2, col_idx], projection='3d')
                ax3d.scatter(x, y, z, s=5, alpha=0.6)
                ax3d.set_title(f"P{p1}S{s1} vs P{p2}S{s2}")
                ax3d.set_xlabel('ΔT corrected (ns)')
                ax3d.set_ylabel('Q_sum_semisum')
                ax3d.set_zlabel('Q_sum_semidiff')

                ax3d.set_xlim([delta_t_left, delta_t_right])
                ax3d.set_ylim([q_sum_left, q_sum_right])
                ax3d.set_zlim([q_diff_left, q_diff_right])

                # XY projection
                ax_xy = fig.add_subplot(spec[2, col_idx])
                ax_xy.scatter(x, y, s=5, alpha=0.5)
                z_fixed = np.mean(z)
                y_line = np.linspace(np.min(y), np.max(y), 100)
                x_line = a * y_line + b * z_fixed + c
                ax_xy.plot(x_line, y_line, color='black', lw=1, label='Fitted projection')
                ax_xy.set_xlabel('ΔT corrected')
                ax_xy.set_ylabel('Q_sum_semisum')
                ax_xy.set_title('XY projection')
                ax_xy.legend(fontsize='x-small')
                ax_xy.set_xlim([delta_t_left, delta_t_right])
                ax_xy.set_ylim([q_sum_left, q_sum_right])

                # XZ projection
                ax_xz = fig.add_subplot(spec[3, col_idx])
                ax_xz.scatter(x, z, s=5, alpha=0.5, c='tab:red')
                y_fixed = np.mean(y)
                z_line = np.linspace(np.min(z), np.max(z), 100)
                x_line2 = a * y_fixed + b * z_line + c
                ax_xz.plot(x_line2, z_line, color='black', lw=1, label='Fitted projection')
                ax_xz.set_xlabel('ΔT corrected')
                ax_xz.set_ylabel('Q_sum_semidiff')
                ax_xz.set_title('XZ projection')
                ax_xz.legend(fontsize='x-small')
                ax_xz.set_xlim([T_sum_corrected_diff_left, T_sum_corrected_diff_right])
                ax_xz.set_ylim([Q_sum_semidiff_left, Q_sum_semidiff_right])

            fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at the top (5%)
            fig.suptitle(f"Batch {batch + 1}/{num_batches} — 3D Slewing + Fitted Projections", fontsize=16)  # increase font size
            
            if save_plots:
                name_of_file = 'slewing_3d_fitproj'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1

                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')

            if show_plots:
                plt.show()
            plt.close()

    
    # -------------------------------------------------------------------
    # FIT VALIDATION with y = x
    # -------------------------------------------------------------------
    
    # if create_essential_plots or create_plots:
    if create_plots:

        pair_labels = [
            (p1, s1, p2, s2)
            for p1 in range(1, 5)
            for s1 in range(1, 5)
            for p2 in range(p1 + 1, 5)
            for s2 in range(1, 5)
        ]

        batch_size = 8
        num_batches = int(np.ceil(len(pair_labels) / batch_size))
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(pair_labels))
            current_pairs = pair_labels[start_idx:end_idx]

            fig, axes = plt.subplots(2, batch_size, figsize=(4 * batch_size, 8), constrained_layout=True)
            axes = np.atleast_2d(axes)

            for col_idx, (p1, s1, p2, s2) in enumerate(current_pairs):
                mask = (
                    (slew_df['plane1'] == p1) & (slew_df['strip1'] == s1) &
                    (slew_df['plane2'] == p2) & (slew_df['strip2'] == s2)
                )
                data = slew_df[mask]

                valid_mask = (
                    (data['Q_sum_semidiff'] != 0) &
                    (data['Q_sum_semisum'] != 0) &
                    (data['T_sum_corrected_diff'] != 0)
                )
                data = data[valid_mask]

                if len(data) < 10:
                    for row in range(2):
                        axes[row, col_idx].set_visible(False)
                    continue

                fit_row = slewing_fit_df[
                    (slewing_fit_df['plane1'] == p1) &
                    (slewing_fit_df['strip1'] == s1) &
                    (slewing_fit_df['plane2'] == p2) &
                    (slewing_fit_df['strip2'] == s2)
                ]
                if fit_row.empty:
                    for row in range(2):
                        axes[row, col_idx].set_visible(False)
                    continue

                a = fit_row['a_semisum'].values[0]
                b = fit_row['b_semidiff'].values[0]
                c = fit_row['c_offset'].values[0]

                qsum = data['Q_sum_semisum'].values
                qdiff = data['Q_sum_semidiff'].values
                t_true = data['T_sum_corrected_diff'].values
                t_pred = a * qsum + b * qdiff + c
                residual = t_true - t_pred
                
                # Filter residuals, remove if out of the range
                residual_range = slewing_residual_range
                cond = (residual > -1*residual_range) & (residual < residual_range)
                residual = residual[cond]
                t_true = t_true[cond]
                t_pred = t_pred[cond]
                
                # Plot predicted vs true with y=x line
                ax0 = axes[0, col_idx]
                ax0.scatter(t_true, t_pred, s=5, alpha=0.6)
                min_val = min(t_true.min(), t_pred.min())
                max_val = max(t_true.max(), t_pred.max())
                ax0.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
                ax0.set_title(f"P{p1}S{s1} vs P{p2}S{s2}")
                ax0.set_xlabel("T_true")
                ax0.set_ylabel("T_predicted")

                ax0.set_xlim([-t_comparison_lim, t_comparison_lim])
                ax0.set_ylim([-t_comparison_lim, t_comparison_lim])

                # Residuals histogram
                ax1 = axes[1, col_idx]
                ax1.hist(t_true, bins=100, alpha=0.7, color='tab:gray', label = "Uncorrected")
                ax1.hist(residual, bins=100, alpha=0.7, color='green', label = "Residuals, same as corrected")
                ax1.set_xlabel("Residuals (ns)")
                ax1.set_ylabel("Counts")
                ax1.set_title("Residual Distribution")
                ax1.set_xlim([-4, 4])

            # Hide unused subplots
            for row in range(2):
                for col in range(len(current_pairs), batch_size):
                    axes[row, col].set_visible(False)

            fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at the top (5%)
            fig.suptitle(f"Batch {batch + 1}/{num_batches} — Fit Check (Predicted vs Real, Residuals)", fontsize=16)  # increase font size
            if save_plots:
                name_of_file = 'model_validation_simple'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1

                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')

            if show_plots:
                plt.show()
            plt.close()


print("----------------------------------------------------------------------")
print("----------------------- Time sum calibration -------------------------")
print("----------------------------------------------------------------------")

if time_calibration:
    
    if slewing_correction == False:
        old_timing_method = True
    
    if old_timing_method:
        # Initialize an empty list to store the resulting matrices for each event
        event_matrices = []
        
        # Iterate over each event (row) in the DataFrame
        for _, row in working_df.iterrows():
            event_matrix = []
            for module in ['T1', 'T2', 'T3', 'T4']:
                # Find the index of the strip with the maximum Q_sum for this module
                Q_sum_cols = [f'{module.replace("T", "Q")}_Q_sum_{i+1}' for i in range(4)]
                Q_sum_values = row[Q_sum_cols].values
                
                if sum(Q_sum_values) == 0:
                    event_matrix.append([0, 0, 0])
                    continue
                
                max_index = np.argmax(Q_sum_values) + 1
                    
                # Get the corresponding T_sum and T_diff for the module and strip
                T_sum_col = f'{module}_T_sum_{max_index}'
                T_diff_col = f'{module}_T_diff_{max_index}'
                T_sum_value = row[T_sum_col]
                T_diff_value = row[T_diff_col]
                
                # Append the row to the event matrix
                event_matrix.append([max_index, T_sum_value, T_diff_value])
            
            # Convert the event matrix to a numpy array and append it to the list of event matrices
            event_matrices.append(np.array(event_matrix))
        
        # Convert the list of event matrices to a 3D numpy array (events x modules x features)
        event_matrices = np.array(event_matrices)
        
        # The old code to do this -----------------------------
        
        yz_big = np.array([[[y, z] for y in y_pos_T[i % 2]] for i, z in enumerate(z_positions)])
        
        def calculate_diff(P_a, s_a, P_b, s_b, ps):
            
            # First position
            x_1 = ps[P_a-1, 1]
            yz_1 = yz_big[P_a-1, s_a-1]
            xyz_1 = np.append(x_1, yz_1)
            
            # Second position
            x_2 = ps[P_b-1, 1]
            yz_2 = yz_big[P_b-1, s_b-1]
            xyz_2 = np.append(x_2, yz_2)
            
            pos_x.append(x_1)
            pos_x.append(x_2)
            
            t_0_1 = ps[P_a-1, 2]
            t_0_2 = ps[P_b-1, 2]
            t_0.append(t_0_1)
            t_0.append(t_0_2)
            
            # Length
            dist = np.sqrt(np.sum((xyz_2 - xyz_1)**2))
            travel_time = dist / muon_speed
            
            v_travel_time.append(travel_time)
            
            # diff = travel_time
            diff = ps[P_b-1, 2] - ps[P_a-1, 2] - travel_time
            # diff = ps[P_b-1, 2] - ps[P_a-1, 2]
            return diff
        
        # Three layers spaced
        P1s1_P4s1 = []; P1s1_P4s2 = []; P1s2_P4s1 = []; P1s2_P4s2 = []; P1s2_P4s3 = []; P1s3_P4s2 = []; P1s3_P4s3 = []; P1s3_P4s4 = []; P1s4_P4s3 = []; P1s4_P4s4 = []; P1s1_P4s3 = []; P1s3_P4s1 = []; P1s2_P4s4 = []; P1s4_P4s2 = []; P1s1_P4s4 = [];

        # Two layers spaced
        P1s1_P3s1 = []; P1s1_P3s2 = []; P1s2_P3s1 = []; P1s2_P3s2 = []; P1s2_P3s3 = []; P1s3_P3s2 = []; P1s3_P3s3 = []; P1s3_P3s4 = []; P1s4_P3s3 = []; P1s4_P3s4 = []; P1s1_P3s3 = []; P1s3_P3s1 = []; P1s2_P3s4 = []; P1s4_P3s2 = []; P1s1_P3s4 = [];
        P2s1_P4s1 = []; P2s1_P4s2 = []; P2s2_P4s1 = []; P2s2_P4s2 = []; P2s2_P4s3 = []; P2s3_P4s2 = []; P2s3_P4s3 = []; P2s3_P4s4 = []; P2s4_P4s3 = []; P2s4_P4s4 = []; P2s1_P4s3 = []; P2s3_P4s1 = []; P2s2_P4s4 = []; P2s4_P4s2 = []; P2s1_P4s4 = [];

        # One layer spaced
        P1s1_P2s1 = []; P1s1_P2s2 = []; P1s2_P2s1 = []; P1s2_P2s2 = []; P1s2_P2s3 = []; P1s3_P2s2 = []; P1s3_P2s3 = []; P1s3_P2s4 = []; P1s4_P2s3 = []; P1s4_P2s4 = []; P1s1_P2s3 = []; P1s3_P2s1 = []; P1s2_P2s4 = []; P1s4_P2s2 = []; P1s1_P2s4 = [];
        P2s1_P3s1 = []; P2s1_P3s2 = []; P2s2_P3s1 = []; P2s2_P3s2 = []; P2s2_P3s3 = []; P2s3_P3s2 = []; P2s3_P3s3 = []; P2s3_P3s4 = []; P2s4_P3s3 = []; P2s4_P3s4 = []; P2s1_P3s3 = []; P2s3_P3s1 = []; P2s2_P3s4 = []; P2s4_P3s2 = []; P2s1_P3s4 = [];
        P3s1_P4s1 = []; P3s1_P4s2 = []; P3s2_P4s1 = []; P3s2_P4s2 = []; P3s2_P4s3 = []; P3s3_P4s2 = []; P3s3_P4s3 = []; P3s3_P4s4 = []; P3s4_P4s3 = []; P3s4_P4s4 = []; P3s1_P4s3 = []; P3s3_P4s1 = []; P3s2_P4s4 = []; P3s4_P4s2 = []; P3s1_P4s4 = [];
        
        pos_x = []
        v_travel_time = []
        t_0 = []
        
        # -----------------------------------------------------------------------------
        # Perform the calculation of a strip vs. the any other one --------------------
        # -----------------------------------------------------------------------------
        
        i = 0
        for event in event_matrices:
            if limit and i >= limit_number:
                break
            if np.all(event[:,0] == 0):
                continue
            
            istrip = event[:, 0]
            t0 = event[:,1] - strip_length / 2 / strip_speed
            x = event[:,2] * strip_speed
            
            ps = np.column_stack(( istrip, x,  t0 ))
            ps[:,2] = ps[:,2] - ps[0,2]
            
            # ---------------------------------------------------------------------
            # Fill the time differences vectors -----------------------------------
            # ---------------------------------------------------------------------
            
            # Three layers spacing ------------------------------------------------
            # P1-P4 ---------------------------------------------------------------
            P_a = 1; P_b = 4
            # Same strips
            s_a = 1; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Adjacent strips
            s_a = 1; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Two separated strips
            s_a = 1; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Three separated strips
            s_a = 1; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            
            # Two layers spacing --------------------------------------------------
            # P1-P3 ---------------------------------------------------------------
            P_a = 1; P_b = 3
            # Same strips
            s_a = 1; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P3s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P3s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P3s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P3s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Adjacent strips
            s_a = 1; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P3s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P3s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P3s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P3s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P3s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P3s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Two separated strips
            s_a = 1; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P3s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P3s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P3s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P3s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Three separated strips
            s_a = 1; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P3s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            
            # P2-P4 ---------------------------------------------------------------
            P_a = 2; P_b = 4
            # Same strips
            s_a = 1; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s1_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s2_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s3_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s4_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Adjacent strips
            s_a = 1; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s1_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s2_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s2_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s3_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s3_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s4_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Two separated strips
            s_a = 1; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s1_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s3_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s2_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s4_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Three separated strips
            s_a = 1; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s1_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            
            # One layer spacing ---------------------------------------------------
            # P3-P4 ---------------------------------------------------------------
            P_a = 3; P_b = 4
            # Same strips
            s_a = 1; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s1_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s2_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s3_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s4_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Adjacent strips
            s_a = 1; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s1_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s2_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s2_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s3_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s3_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s4_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Two separated strips
            s_a = 1; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s1_P4s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s3_P4s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s2_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s4_P4s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Three separated strips
            s_a = 1; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P3s1_P4s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            
            # P1-P2 ---------------------------------------------------------------
            P_a = 1; P_b = 2
            # Same strips
            s_a = 1; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P2s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P2s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P2s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P2s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Adjacent strips
            s_a = 1; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P2s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P2s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P2s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P2s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P2s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P2s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Two separated strips
            s_a = 1; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P2s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s3_P2s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s2_P2s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s4_P2s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Three separated strips
            s_a = 1; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P1s1_P2s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            
            # P2-P3 ---------------------------------------------------------------
            P_a = 2; P_b = 3
            # Same strips
            s_a = 1; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s1_P3s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s2_P3s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s3_P3s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s4_P3s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Adjacent strips
            s_a = 1; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s1_P3s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s2_P3s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s2_P3s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s3_P3s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s3_P3s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s4_P3s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Two separated strips
            s_a = 1; s_b = 3
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s1_P3s3.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 3; s_b = 1
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s3_P3s1.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 2; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s2_P3s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            s_a = 4; s_b = 2
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s4_P3s2.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
            # Three separated strips
            s_a = 1; s_b = 4
            if ps[P_a-1, 0] == s_a and ps[P_b-1, 0] == s_b: P2s1_P3s4.append(calculate_diff(P_a, s_a, P_b, s_b, ps))
                
            i += 1
        
        vectors = [
            P1s1_P3s1, P1s1_P3s2, P1s2_P3s1, P1s2_P3s2, P1s2_P3s3,
            P1s3_P3s2, P1s3_P3s3, P1s3_P3s4, P1s4_P3s3, P1s4_P3s4,
            P1s1_P3s3, P1s3_P3s1, P1s2_P3s4, P1s4_P3s2, P1s1_P3s4,\
                
            P1s1_P4s1, P1s1_P4s2, P1s2_P4s1, P1s2_P4s2, P1s2_P4s3,
            P1s3_P4s2, P1s3_P4s3, P1s3_P4s4, P1s4_P4s3, P1s4_P4s4,
            P1s1_P4s3, P1s3_P4s1, P1s2_P4s4, P1s4_P4s2, P1s1_P4s4,\
                
            P2s1_P4s1, P2s1_P4s2, P2s2_P4s1, P2s2_P4s2, P2s2_P4s3,
            P2s3_P4s2, P2s3_P4s3, P2s3_P4s4, P2s4_P4s3, P2s4_P4s4,
            P2s1_P4s3, P2s3_P4s1, P2s2_P4s4, P2s4_P4s2, P2s1_P4s4,\
                
            P3s1_P4s1, P3s1_P4s2, P3s2_P4s1, P3s2_P4s2, P3s2_P4s3,
            P3s3_P4s2, P3s3_P4s3, P3s3_P4s4, P3s4_P4s3, P3s4_P4s4,
            P3s1_P4s3, P3s3_P4s1, P3s2_P4s4, P3s4_P4s2, P3s1_P4s4,\
                
            P1s1_P2s1, P1s1_P2s2, P1s2_P2s1, P1s2_P2s2, P1s2_P2s3,
            P1s3_P2s2, P1s3_P2s3, P1s3_P2s4, P1s4_P2s3, P1s4_P2s4,
            P1s1_P2s3, P1s3_P2s1, P1s2_P2s4, P1s4_P2s2, P1s1_P2s4,\
                
            P2s1_P3s1, P2s1_P3s2, P2s2_P3s1, P2s2_P3s2, P2s2_P3s3,
            P2s3_P3s2, P2s3_P3s3, P2s3_P3s4, P2s4_P3s3, P2s4_P3s4,
            P2s1_P3s3, P2s3_P3s1, P2s2_P3s4, P2s4_P3s2, P2s1_P3s4
        ]

        if create_plots:
        # if create_plots or create_essential_plots:
        
            # Convert data to numpy arrays and filter
            pos_x = np.array(pos_x)
            pos_x = pos_x[(-200 < pos_x) & (pos_x < 200) & (pos_x != 0)]
            v_travel_time = np.array(v_travel_time)
            v_travel_time = v_travel_time[v_travel_time < 1.6]
            t_0 = np.array(t_0)
            
            
            t_0 = t_0[(-t0_time_cal_lim < t_0) & (t_0 < t0_time_cal_lim)]
            t_0 = t_0[t_0 != 0]
            
            # Prepare a figure with 1x3 subplots
            fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
            
            # Plot histogram for positions (pos_x)
            axs[0].hist(pos_x, bins='auto', alpha=0.6, color='blue')
            axs[0].set_title('Positions')
            axs[0].set_xlabel('Position (units)')
            axs[0].set_ylabel('Frequency')
            
            # Plot histogram for travel time (v_travel_time)
            axs[1].hist(v_travel_time, bins=300, alpha=0.6, color='green')
            axs[1].set_title('Travel Time of a Particle at c')
            axs[1].set_xlabel('T / ns')
            axs[1].set_ylabel('Frequency')
            
            # Plot histogram for T0s (t_0)
            axs[2].hist(t_0, bins='auto', alpha=0.6, color='red')
            axs[2].set_title('T0s')
            axs[2].set_xlabel('T / ns')
            axs[2].set_ylabel('Frequency')
            
            # Show the combined figure
            plt.suptitle('Combined Histograms of Positions, Travel Time, and T0s')
            if save_plots:
                name_of_file = 'positions_travel_time_tzeros'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
            if show_plots: plt.show()
            plt.close()
        
            for i, vector in enumerate(vectors):
                var_name = [name for name, val in globals().items() if val is vector][0]
                if i >= number_of_time_cal_figures: break
                hist_1d(vector, 100, var_name, "T / ns", var_name)
        
        
        print("----------------------------------------------------------------------")
        print("--------------------- Time resolution calculation --------------------")
        print("----------------------------------------------------------------------")

        # Dictionary to store CRT values
        crt_values = {}
        for i, vector in enumerate(vectors):
            var_name = [name for name, val in globals().items() if val is vector][0]
            vdat = np.array(vector)
            if len(vdat) > 1:
                try:
                    vdat = vdat[(vdat > np.quantile(vdat, CRT_gaussian_fit_quantile)) & (vdat < np.quantile(vdat, 1 - CRT_gaussian_fit_quantile))]
                except IndexError:
                    print(f"IndexError encountered for {var_name}, setting CRT to 0")
                    vdat = np.array([0])
            
            CRT = norm.fit(vdat)[1] / np.sqrt(2) if len(vdat) > 0 else 0
            # print(f"CRT for {var_name} is {CRT:.4g}")
            crt_values[f'CRT_{var_name}'] = CRT
        
        # Turn crt_values into a vector
        print("CRT values:", crt_values)
        crt_values = np.array(list(crt_values.values()))
        # print("CRT values:", crt_values)
        Q1, Q3 = np.percentile(crt_values, [25, 75])
        crt_values = crt_values[crt_values <= 1]
        filtered_crt_values = crt_values[(crt_values >= Q1 - 1.5 * (Q3 - Q1)) & (crt_values <= Q3 + 1.5 * (Q3 - Q1))]
        global_variables['CRT_avg'] = np.mean(filtered_crt_values)*1000 # To ps
        
        print("---------------------------")
        print(f"CRT Avg: {global_variables['CRT_avg']:.4g} ps")
        print("---------------------------")
        
        
        # Create row and column indices
        rows = ['P{}s{}'.format(i, j) for i in range(1, 5) for j in range(1, 5)]
        columns = ['P{}s{}'.format(i, j) for i in range(1, 5) for j in range(1, 5)]
        
        df = pd.DataFrame(index=rows, columns=columns)
        for vector in vectors:
            var_name = [name for name, val in globals().items() if val is vector][0]
            if var_name == "vector":
                continue
            current_prefix = str(var_name.split('_')[0])
            current_suffix = str(var_name.split('_')[1])
            # Key part: create the antisymmetric matrix
            df.loc[current_prefix, current_suffix] = summary(vector)
            df.loc[current_suffix, current_prefix] = -df.loc[current_prefix, current_suffix]
            
    else:
        # Create row and column indices
        rows = ['P{}s{}'.format(i, j) for i in range(1, 5) for j in range(1, 5)]
        columns = ['P{}s{}'.format(i, j) for i in range(1, 5) for j in range(1, 5)]
        
        df = pd.DataFrame(index=rows, columns=columns)
        
        # Fill df with antisymmetric c_offset values from slewing_fit_df
        for _, row in slewing_fit_df.iterrows():
            l1 = f'P{str(int(row["plane1"]))}s{str(int(row["strip1"]))}'
            l2 = f'P{str(int(row["plane2"]))}s{str(int(row["strip2"]))}'
            offset = row['c_offset']
            df.loc[l1, l2] = -offset
            df.loc[l2, l1] = offset
        
    print("Antisymmetric matrix:")
    print(df.map(lambda x: f"{x:.2f}" if pd.notnull(x) else ""))

    
    # -----------------------------------------------------------------------------
    # Brute force method ----------------------------------------------------------
    # -----------------------------------------------------------------------------
    
    if brute_force_analysis_time_calibration_path_finding:
        # Main itinerary
        itinerary = ["P1s1", "P3s1", "P1s2", "P3s2", "P1s3", "P3s3", "P1s4", "P3s4", "P4s4", "P2s4", "P4s3", "P2s3", "P4s2", "P2s2", "P4s1", "P2s1"]
        k = 0
        max_iter = 2000000
        brute_force_list = []
        # Create row and column indices
        rows = ['P{}'.format(i) for i in range(1, 5)]
        columns = ['s{}'.format(i) for i in range(1,5)]
        brute_force_df = pd.DataFrame(0, index=rows, columns=columns)
        jump = False

        existing_itineraries = load_itineraries_from_file(ITINERARY_FILE_PATH, required=False)
        successful_itineraries: dict[tuple[str, ...], None] = {
            tuple(existing_itinerary): None
            for existing_itinerary in existing_itineraries
        }
        found_new_itinerary = False

        while k < max_iter:
            if k % 50000 == 0: print(f"Itinerary {k}")
            brute_force_df[brute_force_df.columns] = 0
            step = itinerary
            a = []
            for i in range(len(itinerary)):
                if i > 0:
                    # Storing new values
                    a.append( df[step[i - 1]][step[i]] )
                relative_time = sum(a)
                if np.isnan(relative_time):
                    jump = True
                    break
                ind1 = str(step[i][0:2])
                ind2 = str(step[i][2:4])
                brute_force_df.loc[ind1,ind2] = brute_force_df.loc[ind1,ind2] + relative_time
            # If the path is succesful, print it, then we can copy it from terminal
            # and save it for the next step.
            if jump == False:
                print(itinerary)
                itinerary_tuple = tuple(step)
                if itinerary_tuple not in successful_itineraries:
                    successful_itineraries[itinerary_tuple] = None
                    found_new_itinerary = True
            # Shuffle the path
            random.shuffle(itinerary)
            # Iterate
            k += 1
            if jump:
                jump = False
                continue
            # Substract a value from the entire DataFrame
            brute_force_df = brute_force_df.sub(brute_force_df.iloc[0, 0])
            # Append the matrix to the big list
            brute_force_list.append(brute_force_df.values)
        # Calculate the mean of all the paths
        calibrated_times_bf = np.nanmean(brute_force_list, axis=0)
        calibration_times = calibrated_times_bf

        if successful_itineraries and (found_new_itinerary or not ITINERARY_FILE_PATH.exists()):
            write_itineraries_to_file(ITINERARY_FILE_PATH, successful_itineraries.keys())
    
    
    # -----------------------------------------------------------------------------
    # Selected paths method
    # -----------------------------------------------------------------------------
    try:
        itineraries = load_itineraries_from_file(ITINERARY_FILE_PATH)
    except (FileNotFoundError, ValueError) as itinerary_error:
        print(itinerary_error)
        sys.exit(1)

    def has_duplicate_sublists(lst):
        seen = set()
        for sub_list in lst:
            sub_list_tuple = tuple(sub_list)
            if sub_list_tuple in seen:
                return True
            seen.add(sub_list_tuple)
        return False
    
    if has_duplicate_sublists(itineraries):
        print("Duplicated itineraries.")
    
    selected_path_list = []
    rows = ['P{}'.format(i) for i in range(1, 5)]
    columns = ['s{}'.format(i) for i in range(1,5)]
    selected_path_df = pd.DataFrame(0, index=rows, columns=columns)
    
    for itinerary in itineraries:
        selected_path_df[selected_path_df.columns] = 0
        step = itinerary
        a = []
        for i in range(len(step)):
            if i > 0:
                a.append( df[step[i - 1]][step[i]] )
            
            relative_time = sum(a)
            ind1 = str(step[i][0:2])
            ind2 = str(step[i][2:4])
            
            selected_path_df[ind2] = selected_path_df[ind2].astype(float)
            # selected_path_df.loc[ind1,ind2] = selected_path_df.loc[ind1,ind2] - relative_time
            selected_path_df.loc[ind1,ind2] = selected_path_df.loc[ind1,ind2] + relative_time # ORIGINALLY THERE WAS A MINUS BUT STOPPED WORKING SO I PUT THE + TO TRY
        
        selected_path_df = selected_path_df.sub(selected_path_df.iloc[0, 0])
        selected_path_list.append(selected_path_df.values)
        
    # Calculate the mean of all the paths
    # calibration_times = np.nanmean(selected_path_list, axis=0)
    
    # selected_path_list: shape (N_paths, N_points)
    selected_path_array = np.array(selected_path_list)
    median = np.nanmedian(selected_path_array, axis=0)
    abs_dev = np.abs(selected_path_array - median)
    epsilon = 1e-6
    weights = 1.0 / (abs_dev + epsilon)
    weighted_sum = np.nansum(selected_path_array * weights, axis=0)
    sum_of_weights = np.nansum(weights, axis=0)
    calibration_times = weighted_sum / sum_of_weights
    
    # Time calibration matrix calculated --------------------------------------
    
    print("------------------------")
    print("Calibration in times is:\n", calibration_times)
    
    diff = np.abs(calibration_times - time_sum_reference) > time_sum_distance
    nan_mask = np.isnan(calibration_times)
    values_replaced_t_sum = np.any(diff | nan_mask)
    calibration_times[diff | nan_mask] = time_sum_reference[diff | nan_mask]
    if values_replaced_t_sum:
        print("Some values were replaced in the calibration in times.")
    
    # Applying time calibration
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            mask = working_df[f'{key}_T_sum_{j+1}'] != 0
            working_df.loc[mask, f'{key}_T_sum_{j+1}'] += calibration_times[i][j]
    
else:
    calibration_times = time_sum_reference
    working_df['CRT_avg'] = 1000 # An extreme time to not crush the program
    print("Calibration in times was set to the reference! (calibration was not performed)\n", calibration_times)
    
    # Applying time calibration
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            mask = working_df[f'{key}_T_sum_{j+1}'] != 0
            working_df.loc[mask, f'{key}_T_sum_{j+1}'] += calibration_times[i][j]



print("----------------------------------------------------------------------")
print("--------------- Cross-talk filtering, will be set to 0 ---------------")
print("----------------------------------------------------------------------")

crosstalk_removal_and_recalibration = True

if crosstalk_removal_and_recalibration:

    crosstalk_pedestal = {
        "crstlk_pedestal_P1s1": 0, "crstlk_pedestal_P1s2": 0, "crstlk_pedestal_P1s3": 0, "crstlk_pedestal_P1s4": 0,
        "crstlk_pedestal_P2s1": 0, "crstlk_pedestal_P2s2": 0, "crstlk_pedestal_P2s3": 0, "crstlk_pedestal_P2s4": 0,
        "crstlk_pedestal_P3s1": 0, "crstlk_pedestal_P3s2": 0, "crstlk_pedestal_P3s3": 0, "crstlk_pedestal_P3s4": 0,
        "crstlk_pedestal_P4s1": 0, "crstlk_pedestal_P4s2": 0, "crstlk_pedestal_P4s3": 0, "crstlk_pedestal_P4s4": 0
    }

    crosstalk_limits = {
        "crstlk_limit_P1s1": 0, "crstlk_limit_P1s2": 0, "crstlk_limit_P1s3": 0, "crstlk_limit_P1s4": 0,
        "crstlk_limit_P2s1": 0, "crstlk_limit_P2s2": 0, "crstlk_limit_P2s3": 0, "crstlk_limit_P2s4": 0,
        "crstlk_limit_P3s1": 0, "crstlk_limit_P3s2": 0, "crstlk_limit_P3s3": 0, "crstlk_limit_P3s4": 0,
        "crstlk_limit_P4s1": 0, "crstlk_limit_P4s2": 0, "crstlk_limit_P4s3": 0, "crstlk_limit_P4s4": 0
    }
    
    crosstalk_mean = {
        "crstlk_mu_P1s1": 0, "crstlk_mu_P1s2": 0, "crstlk_mu_P1s3": 0, "crstlk_mu_P1s4": 0,
        "crstlk_mu_P2s1": 0, "crstlk_mu_P2s2": 0, "crstlk_mu_P2s3": 0, "crstlk_mu_P2s4": 0,
        "crstlk_mu_P3s1": 0, "crstlk_mu_P3s2": 0, "crstlk_mu_P3s3": 0, "crstlk_mu_P3s4": 0,
        "crstlk_mu_P4s1": 0, "crstlk_mu_P4s2": 0, "crstlk_mu_P4s3": 0, "crstlk_mu_P4s4": 0
    }
    
    crosstalk_std = {
        "crstlk_sigma_P1s1": 0, "crstlk_sigma_P1s2": 0, "crstlk_sigma_P1s3": 0, "crstlk_sigma_P1s4": 0,
        "crstlk_sigma_P2s1": 0, "crstlk_sigma_P2s2": 0, "crstlk_sigma_P2s3": 0, "crstlk_sigma_P2s4": 0,
        "crstlk_sigma_P3s1": 0, "crstlk_sigma_P3s2": 0, "crstlk_sigma_P3s3": 0, "crstlk_sigma_P3s4": 0,
        "crstlk_sigma_P4s1": 0, "crstlk_sigma_P4s2": 0, "crstlk_sigma_P4s3": 0, "crstlk_sigma_P4s4": 0
    }
    
    crosstalk_ampl = {
        "crstlk_ampl_P1s1": 0, "crstlk_ampl_P1s2": 0, "crstlk_ampl_P1s3": 0, "crstlk_ampl_P1s4": 0,
        "crstlk_ampl_P2s1": 0, "crstlk_ampl_P2s2": 0, "crstlk_ampl_P2s3": 0, "crstlk_ampl_P2s4": 0,
        "crstlk_ampl_P3s1": 0, "crstlk_ampl_P3s2": 0, "crstlk_ampl_P3s3": 0, "crstlk_ampl_P3s4": 0,
        "crstlk_ampl_P4s1": 0, "crstlk_ampl_P4s2": 0, "crstlk_ampl_P4s3": 0, "crstlk_ampl_P4s4": 0
    }
    
    crosstalk_linear = {
        "crstlk_mx_b_P1s1": [0, 0], "crstlk_mx_b_P1s2": [0, 0], "crstlk_mx_b_P1s3": [0, 0], "crstlk_mx_b_P1s4": [0, 0],
        "crstlk_mx_b_P2s1": [0, 0], "crstlk_mx_b_P2s2": [0, 0], "crstlk_mx_b_P2s3": [0, 0], "crstlk_mx_b_P2s4": [0, 0],
        "crstlk_mx_b_P3s1": [0, 0], "crstlk_mx_b_P3s2": [0, 0], "crstlk_mx_b_P3s3": [0, 0], "crstlk_mx_b_P3s4": [0, 0],
        "crstlk_mx_b_P4s1": [0, 0], "crstlk_mx_b_P4s2": [0, 0], "crstlk_mx_b_P4s3": [0, 0], "crstlk_mx_b_P4s4": [0, 0]
    }

    # Gaussian + linear function
    def gaussian_linear(x, a, mu, sigma, m, b):
        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + m * x + b
    
    for i, key in enumerate(['1', '2', '3', '4']):
        for j in range(4):
            col = f'Q{key}_Q_sum_{j+1}'
            y = working_df[col]
            
            Q_clip_min = pedestal_left
            Q_clip_max = pedestal_right
            
            num_bins = 80
            data = y[(y != 0) & (y > Q_clip_min) & (y < Q_clip_max)]
            
            hist_vals, bin_edges = np.histogram(data, bins=num_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            try:
                a_min = 0
                a_max = 2*max(hist_vals) + 1
                
                mu_min = pedestal_left # -1
                mu_max = crosstalk_fit_mu_max
                
                sigma_min = crosstalk_fit_sigma_min
                sigma_max = crosstalk_fit_sigma_max
                
                # print(f"P{key}s{j+1}")
                
                popt, _ = curve_fit(
                    gaussian_linear, 
                    bin_centers, 
                    hist_vals, 
                    p0=[max(hist_vals), 0, 1, 0, min(hist_vals)], 
                    bounds=([a_min, mu_min, sigma_min, -np.inf, -np.inf], [a_max, mu_max, sigma_max, np.inf, np.inf])
                )
                
                a, mu, sigma, m, b = popt
                
                crosstalk_ampl[f'crstlk_ampl_P{key}s{j+1}'] = a
                crosstalk_mean[f'crstlk_mu_P{key}s{j+1}'] = mu
                crosstalk_std[f'crstlk_sigma_P{key}s{j+1}'] = sigma
                crosstalk_linear[f'crstlk_mx_b_P{key}s{j+1}'] = [m, b]
                
                crosstalk_pedestal[f'crstlk_pedestal_P{key}s{j+1}'] = mu - 2 * sigma
                crosstalk_limits[f'crstlk_limit_P{key}s{j+1}'] = min([mu + 3 * sigma, 1.2])
                
            except RuntimeError:
                continue
    
    print("\nCrosstalk limit after fitting a gaussian to the peak:")
    values = [crosstalk_limits[f"crstlk_limit_P{p}s{s}"] for p in range(1, 5) for s in range(1, 5)]
    matrix = np.array(values).reshape(4, 4)
    print(matrix, '\n')
    
    # if create_plots:
    if create_plots or create_essential_plots:
        fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_Q = axes_Q.flatten()

        for i, key in enumerate(['1', '2', '3', '4']):
            for j in range(4):
                col = f'Q{key}_Q_sum_{j+1}'
                y = working_df[col]
                
                Q_plot_min = 0.8 * Q_clip_min
                Q_plot_max = 1.4 * Q_clip_max
                
                data = y[(y != 0) & (y > Q_plot_min) & (y < Q_plot_max)]
                
                hist_vals, bin_edges = np.histogram(data, bins=num_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                axes_Q[i*4 + j].axvline(crosstalk_pedestal[f'crstlk_pedestal_P{key}s{j+1}'], color='blue', linestyle='--', alpha=0.5)
                axes_Q[i*4 + j].axvline(crosstalk_limits[f'crstlk_limit_P{key}s{j+1}'], color='blue', linestyle='--', alpha=0.5)
                
                a = crosstalk_ampl[f'crstlk_ampl_P{key}s{j+1}']
                mu = crosstalk_mean[f'crstlk_mu_P{key}s{j+1}']
                sigma = crosstalk_std[f'crstlk_sigma_P{key}s{j+1}']
                m, b = crosstalk_linear[f'crstlk_mx_b_P{key}s{j+1}']
                
                popt = a, mu, sigma, m, b
                
                x_fit = np.linspace(Q_plot_min, Q_plot_max, 500)
                y_fit = gaussian_linear(x_fit, *popt)
                axes_Q[i*4 + j].plot(x_fit, y_fit, 'r--', label='Gauss + Linear Fit')
                
                axes_Q[i*4 + j].hist(data, bins=num_bins, alpha=0.5, label=f'{col}')
                axes_Q[i*4 + j].set_title(f'{col}')
                axes_Q[i*4 + j].legend()
                axes_Q[i*4 + j].set_xlim([Q_plot_min, Q_plot_max])
                axes_Q[i*4 + j].set_ylim([0, None])
                axes_Q[i*4 + j].axvline(0, color='green', linestyle='--', alpha=0.5)
                
        # Display a vertical green dashed, alpha = 0.5 line at 0
        for ax in axes_Q:
            ax.axvline(0, color='green', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Cross-talk study for filtering (zoom), mingo0{station}\n{start_time}", fontsize=16)
        if save_plots:
            final_filename = f'{fig_idx}_cross_talk_filtering_zoom.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close(fig_Q)
    

    print("----------------------------------------------------------------------")
    print("-------------- Filter 5: charge sum crosstalk filtering --------------")
    print("----------------------------------------------------------------------")
    
    new_columns = {}
    for i, key in enumerate(['1', '2', '3', '4']):
        for j in range(4):
            col_name = f'Q{key}_Q_sum_{j+1}'
            if col_name in working_df.columns:
                new_col_name = f'{col_name}_with_crstlk'
                new_new_col_name = f'{col_name}_no_crstlk'
                original_col = working_df[col_name]
                new_columns[new_col_name] = original_col.copy()
                new_columns[new_new_col_name] = original_col.copy()
                new_columns[new_new_col_name] = np.where( original_col < crosstalk_limits[f'crstlk_limit_P{key}s{j+1}'], 0, original_col )

    if new_columns:
        working_df = pd.concat([working_df, pd.DataFrame(new_columns)], axis=1)

    working_df = working_df.copy()
    
    if self_trigger:
        new_columns = {}
        for i, key in enumerate(['1', '2', '3', '4']):
            for j in range(4):
                col_name = f'Q{key}_Q_sum_{j+1}'
                if col_name in working_st_df.columns:
                    new_col_name = f'{col_name}_with_crstlk'
                    new_new_col_name = f'{col_name}_no_crstlk'
                    original_col = working_df_st[col_name]
                    new_columns[new_col_name] = original_col.copy()
                    new_columns[new_new_col_name] = original_col.copy()
                    new_columns[new_new_col_name] = np.where( original_col < crosstalk_limits[f'crstlk_limit_P{key}s{j+1}'], 0, original_col )

        if new_columns:
            working_st_df = pd.concat([working_st_df, pd.DataFrame(new_columns)], axis=1)

        working_st_df = working_st_df.copy()
    
    
    if create_plots or create_essential_plots:
    # if create_plots:
        fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_Q = axes_Q.flatten()

        for i, key in enumerate(['1', '2', '3', '4']):
            for j in range(4):
                col = f'Q{key}_Q_sum_{j+1}'
                y = working_df[col]
                
                Q_clip_min = pedestal_left
                Q_clip_max = pedestal_right * 10
                
                num_bins = 80
                data = y[(y != 0) & (y > Q_clip_min) & (y < Q_clip_max)]
                
                axes_Q[i*4 + j].hist(data, bins=num_bins, alpha=0.5, label=f'{col}')
                axes_Q[i*4 + j].set_title(f'{col}')
                axes_Q[i*4 + j].legend()
                axes_Q[i*4 + j].set_xlim([Q_clip_min, Q_clip_max])
                axes_Q[i*4 + j].set_ylim([0, None])
                axes_Q[i*4 + j].axvline(0, color='green', linestyle='--', alpha=0.5)
                
        # Display a vertical green dashed, alpha = 0.5 line at 0
        for ax in axes_Q:
            ax.axvline(0, color='green', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Cross-talk check for filtering (zoom), mingo0{station}\n{start_time}", fontsize=16)
        if save_plots:
            final_filename = f'{fig_idx}_cross_talk_filtering_zoom_check_no_subs_pedestal.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close(fig_Q)
        
    
    print("----------------------------------------------------------------------")
    print("------------------- Crosstalk pedestal recalibration -----------------")
    print("----------------------------------------------------------------------")
    
    # Apply the pedestal recalibration
    for i, key in enumerate(['1', '2', '3', '4']):
        for j in range(4):
            mask = working_df[f'Q{key}_Q_sum_{j+1}'] != 0
            working_df.loc[mask, f'Q{key}_Q_sum_{j+1}'] -= crosstalk_pedestal[f'crstlk_pedestal_P{key}s{j+1}']
    
    
    if self_trigger:
        for i, key in enumerate(['1', '2', '3', '4']):
            for j in range(4):
                mask = working_st_df[f'Q{key}_Q_sum_{j+1}'] != 0
                working_st_df.loc[mask, f'Q{key}_Q_sum_{j+1}'] -= crosstalk_pedestal[f'crstlk_pedestal_P{key}s{j+1}']


    # if create_plots or create_essential_plots:
    if create_plots:
        fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_Q = axes_Q.flatten()

        for i, key in enumerate(['1', '2', '3', '4']):
            for j in range(4):
                col = f'Q{key}_Q_sum_{j+1}'
                y = working_df[col]
                
                Q_clip_min = pedestal_left
                Q_clip_max = pedestal_right * 1.4
                
                num_bins = 80
                data = y[(y != 0) & (y > Q_clip_min) & (y < Q_clip_max)]
                
                axes_Q[i*4 + j].hist(data, bins=num_bins, alpha=0.5, label=f'{col}')
                axes_Q[i*4 + j].set_title(f'{col}')
                axes_Q[i*4 + j].legend()
                axes_Q[i*4 + j].set_xlim([Q_clip_min, Q_clip_max])
                axes_Q[i*4 + j].set_ylim([0, None])
                axes_Q[i*4 + j].axvline(0, color='green', linestyle='--', alpha=0.5)
                
        # Display a vertical green dashed, alpha = 0.5 line at 0
        for ax in axes_Q:
            ax.axvline(0, color='green', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Cross-talk check for filtering (zoom), mingo0{station}\n{start_time}", fontsize=16)
        if save_plots:
            final_filename = f'{fig_idx}_cross_talk_filtering_zoom_check.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close(fig_Q)


print("----------------------------------------------------------------------")
print("---------- Filter if any variable in the strip is 0 (2/3) ------------")
print("----------------------------------------------------------------------")

# Now go throuhg every plane and strip and if any of the T_sum, T_diff, Q_sum, Q_diff == 0,
# put the four variables in that plane, strip and event to 0

total_events = len(working_df)

for plane in range(1, 5):
    for strip in range(1, 5):
        q_sum  = f'Q{plane}_Q_sum_{strip}'
        q_diff = f'Q{plane}_Q_diff_{strip}'
        t_sum  = f'T{plane}_T_sum_{strip}'
        t_diff = f'T{plane}_T_diff_{strip}'
        
        # Build mask
        mask = (
            (working_df[q_sum]  == 0) |
            (working_df[q_diff] == 0) |
            (working_df[t_sum]  == 0) |
            (working_df[t_diff] == 0)
        )
        
        # Count affected events
        num_affected_events = mask.sum()
        print(f"Plane {plane}, Strip {strip}: {num_affected_events} out of {total_events} events affected ({(num_affected_events / total_events) * 100:.2f}%)")

        # Zero the affected values
        working_df.loc[mask, [q_sum, q_diff, t_sum, t_diff]] = 0




# Print all the column names in the dataframe
print("Columns in the dataframe:")
for col in working_df.columns:
    print(col)
    # If the row contains _F_ or _B_, remove it from the dataframe
    if '_F_' in col or '_B_' in col:
        working_df.drop(columns=[col], inplace=True)
        print(f"Column {col} removed from the dataframe.")




slewing_correction = False
if slewing_correction:
    
    print("----------------------------------------------------------------------")
    print("----------------------- Slewing correction 2/2 -----------------------")
    print("----------------------------------------------------------------------")
    
    # if create_essential_plots or create_plots:
    if create_plots:
        
        plt.figure(figsize=(8, 5))
        plt.hist(slewing_fit_df["r2_score"], bins=20, alpha=0.7)
        plt.xlabel("R² Score")
        plt.ylabel("Frequency")
        plt.title("R² Scores from Slewing Fits")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        if save_plots:
            name_of_file = 'r2_scores'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1

            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')

        if show_plots: plt.show()
        plt.close()
    
    # Step 1: Select pairs with good fit quality
    
    r2_threshold = slewing_correction_r2_threshold
    good_fits_df = slewing_fit_df[slewing_fit_df["r2_score"] > r2_threshold].copy()

    # Store good fit keys
    good_fit_keys = [
        ((row["plane1"], row["strip1"]), (row["plane2"], row["strip2"]))
        for _, row in good_fits_df.iterrows()
    ]

    # Step 2: Process event-by-event and apply correction collectively
    tsum_cols = [col for col in working_df.columns if col.startswith('T') and 'T_sum' in col and '_final' not in col]
    qsum_cols = [col for col in working_df.columns if col.startswith('Q') and 'Q_sum' in col and '_final' not in col]
    ps_labels = [(int(col[1]), int(col[-1])) for col in tsum_cols]

    # Backup original data for plotting comparison later
    working_df_uncorrected = working_df[tsum_cols].copy()

    # Create dictionary of valid fit parameters for fast lookup
    fit_dict = {}

    for _, row in slewing_fit_df.iterrows():
        if row['r2_score'] > r2_threshold:
            key = (int(row['plane1']), int(row['strip1']), int(row['plane2']), int(row['strip2']))
            fit_dict[key] = (row['a_semisum'], row['b_semidiff'], row['c_offset'])


    # Apply slewing correction event-by-event
    for idx, row in working_df.iterrows():
        # Identify strips with valid T and Q in this event
        present_ps = [(p, s) for p, s in ps_labels if row.get(f"T{p}_T_sum_{s}", 0) != 0 and row.get(f"Q{p}_Q_sum_{s}", 0) != 0]
        
        if len(present_ps) < 2:
            continue

        # Find which of the good fit pairs are present in this event
        valid_pairs_in_event = []
        for (ps1, ps2) in good_fit_keys:
            if ps1 in present_ps and ps2 in present_ps:
                valid_pairs_in_event.append((ps1, ps2))

        if not valid_pairs_in_event:
            continue

        # Store multiple corrected tsum estimates
        corrected_deltas_by_ps = {ps: [] for ps in present_ps}

        # Loop over all valid pairs and apply correction symmetrically
        for (p1s, p2s) in valid_pairs_in_event:
            p1, s1 = p1s
            p2, s2 = p2s

            t1 = row[f"T{int(p1)}_T_sum_{int(s1)}"]
            t2 = row[f"T{int(p2)}_T_sum_{int(s2)}"]
            q1 = row[f"Q{int(p1)}_Q_sum_{int(s1)}"]
            q2 = row[f"Q{int(p2)}_Q_sum_{int(s2)}"]

            key_forward = (p1, s1, p2, s2)
            key_reverse = (p2, s2, p1, s1)

            if key_forward in fit_dict:
                a, b, c = fit_dict[key_forward]
                delta_measured = t1 - t2
                delta_q = 0.5 * (q1 - q2)
                correction = a * 0.5 * (q1 + q2) + b * delta_q
                corrected_diff = delta_measured - correction
                # Reconstruct both
                corrected_deltas_by_ps[(p1, s1)].append(corrected_diff / 2)
                corrected_deltas_by_ps[(p2, s2)].append(-corrected_diff / 2)

            elif key_reverse in fit_dict:
                a, b, c = fit_dict[key_reverse]
                delta_measured = t2 - t1
                delta_q = 0.5 * (q2 - q1)
                correction = a * 0.5 * (q2 + q1) + b * delta_q
                corrected_diff = delta_measured - correction
                corrected_deltas_by_ps[(p2, s2)].append(corrected_diff / 2)
                corrected_deltas_by_ps[(p1, s1)].append(-corrected_diff / 2)

        # Compute average corrected value for each T_sum and apply
        mean_correction = np.mean([np.mean(v) for v in corrected_deltas_by_ps.values() if v])
        for (p, s), values in corrected_deltas_by_ps.items():
            if values:
                corrected_value = mean_correction + np.mean(values)
                working_df.at[idx, f"T{p}_T_sum_{s}"] = corrected_value



if create_essential_plots or create_plots:
# if create_plots:

    # Select only the columns that have 'Q_sum', 'Q_diff', 'T_sum', or 'T_diff' in their names
    plot_df = working_df.copy()
    plot_df = plot_df[[col for col in plot_df.columns if any(x in col for x in ['Q_sum', 'Q_diff', 'T_sum', 'T_diff'])]]
    
    num_columns = len(plot_df.columns) - 1  # Exclude 'datetime'
    num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
    fig, axes = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2))
    axes = axes.flatten()

    for i, col in enumerate([col for col in plot_df.columns if col != 'datetime']):
        y = plot_df[col]
        
        if 'Q_sum' in col:
            color = Q_sum_color
        elif 'Q_diff' in col:
            color = Q_diff_color
        elif 'T_sum' in col:
            color = T_sum_color
        elif 'T_diff' in col:
            color = T_diff_color
        else:
            print(col)
            continue
        axes[i].hist(y[y != 0], bins=100, alpha=0.5, label=col, color=color)
        axes[i].set_title(col)
        # axes[i].legend()
        if 'Q_sum' in col:
            axes[i].set_yscale('log')
    
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at the top (5%)
    fig.suptitle("Calibrated filtered data including time calibration removing zeroes in any variable", fontsize=20)  # increase font size
    if save_plots:
        name_of_file = 'time_calibrated_filtered_removed_zeroes'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: 
        plt.show()
    plt.close()



# print("----------------------------------------------------------------------")
# print("----- Time sum setting reference to the upper-most non-zero strip -----")
# print("----------------------------------------------------------------------")

# # Take the max non-zero value of tsum and substract it from all the tsum values in row
# for idx, row in working_df.iterrows():
#     tsum_values = []
#     for plane in range(1, 5):
#         for strip in range(1, 5):
#             t_sum_col = f'T{plane}_T_sum_{strip}'
#             value = row[t_sum_col]
#             if value != 0:
#                 tsum_values.append(value)
    
#     if tsum_values:
#         max_tsum = tsum_values[0]
#         for plane in range(1, 5):
#             for strip in range(1, 5):
#                 t_sum_col = f'T{plane}_T_sum_{strip}'
#                 if row[t_sum_col] != 0:
#                     working_df.at[idx, t_sum_col] = row[t_sum_col] - max_tsum



# if create_essential_plots or create_plots:
# # if create_plots:

#     # Select only the columns that have 'Q_sum', 'Q_diff', 'T_sum', or 'T_diff' in their names
#     plot_df = working_df.copy()
#     plot_df = plot_df[[col for col in plot_df.columns if any(x in col for x in ['Q_sum', 'Q_diff', 'T_sum', 'T_diff'])]]
    
#     num_columns = len(plot_df.columns) - 1  # Exclude 'datetime'
#     num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
#     fig, axes = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2))
#     axes = axes.flatten()

#     for i, col in enumerate([col for col in plot_df.columns if col != 'datetime']):
#         y = plot_df[col]
        
#         if 'Q_sum' in col:
#             color = Q_sum_color
#         elif 'Q_diff' in col:
#             color = Q_diff_color
#         elif 'T_sum' in col:
#             color = T_sum_color
#         elif 'T_diff' in col:
#             color = T_diff_color
#         else:
#             print(col)
#             continue
#         axes[i].hist(y[y != 0], bins=100, alpha=0.5, label=col, color=color)
#         axes[i].set_title(col)
#         # axes[i].legend()
#         if 'Q_sum' in col:
#             axes[i].set_yscale('log')
    
#     # Remove any unused axes
#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])
    
#     fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space at the top (5%)
#     fig.suptitle("Calibrated filtered data including time calibration and referencing removing zeroes in any variable", fontsize=20)  # increase font size
#     if save_plots:
#         name_of_file = 'time_calibrated_filtered_removed_zeroes'
#         final_filename = f'{fig_idx}_{name_of_file}.png'
#         fig_idx += 1

#         save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
#         plot_list.append(save_fig_path)
#         plt.savefig(save_fig_path, format='png')
    
#     if show_plots: 
#         plt.show()
#     plt.close()



# print("----------------------------------------------------------------------")
# print("------------------------- Time sum filtering -------------------------")
# print("----------------------------------------------------------------------")

# for col in working_df.columns:
#     if '_T_sum_' in col:
#         working_df[col] = np.where((working_df[col] > T_sum_right_cal) | (working_df[col] < T_sum_left_cal), 0, working_df[col])


# print("----------------------------------------------------------------------")
# print("---------- Filter if any variable in the strip is 0 (3/3) ------------")
# print("----------------------------------------------------------------------")

# # Now go throuhg every plane and strip and if any of the T_sum, T_diff, Q_sum, Q_diff == 0,
# # put the four variables in that plane, strip and event to 0

# total_events = len(working_df)

# for plane in range(1, 5):
#     for strip in range(1, 5):
#         q_sum  = f'Q{plane}_Q_sum_{strip}'
#         q_diff = f'Q{plane}_Q_diff_{strip}'
#         t_sum  = f'T{plane}_T_sum_{strip}'
#         t_diff = f'T{plane}_T_diff_{strip}'
        
#         # Build mask
#         mask = (
#             (working_df[q_sum]  == 0) |
#             (working_df[q_diff] == 0) |
#             (working_df[t_sum]  == 0) |
#             (working_df[t_diff] == 0)
#         )
        
#         # Count affected events
#         num_affected_events = mask.sum()
#         print(f"Plane {plane}, Strip {strip}: {num_affected_events} out of {total_events} events affected ({(num_affected_events / total_events) * 100:.2f}%)")

#         # Zero the affected values
#         working_df.loc[mask, [q_sum, q_diff, t_sum, t_diff]] = 0


# if self_trigger:
#     total_events = len(working_st_df)

#     for plane in range(1, 5):
#         for strip in range(1, 5):
#             q_sum  = f'Q{plane}_Q_sum_{strip}'
#             q_diff = f'Q{plane}_Q_diff_{strip}'
#             t_sum  = f'T{plane}_T_sum_{strip}'
#             t_diff = f'T{plane}_T_diff_{strip}'
        
#             # Build mask
#             mask = (
#                 (working_st_df[q_sum]  == 0) |
#                 (working_st_df[q_diff] == 0) |
#                 (working_st_df[t_sum]  == 0) |
#                 (working_st_df[t_diff] == 0)
#             )
        
#             # Count affected events
#             num_affected_events = mask.sum()
#             print(f"Plane {plane}, Strip {strip}: {num_affected_events} out of {total_events} events affected ({(num_affected_events / total_events) * 100:.2f}%)")

#             # Zero the affected values
#             working_st_df.loc[mask, [q_sum, q_diff, t_sum, t_diff]] = 0


# print("----------------------------------------------------------------------")
# print("----------------- Filter the Tsum values in a gaussian ---------------")
# print("----------------------------------------------------------------------")

# # Define the sum of two Gaussians
# def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
#     return (A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) +
#             A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2)))



# def find_true_max(A1, mu1, sigma1, A2, mu2, sigma2):
#     # Initial guess: midpoint between the two peaks
#     x0 = (mu1 + mu2) / 2
    
#     # Search bounds: wider range around the peaks
#     bounds = (min(mu1 - 3*sigma1, mu2 - 3*sigma2), 
#               max(mu1 + 3*sigma1, mu2 + 3*sigma2))
    
#     # Find the maximum by minimizing the negative of the function
#     result = minimize_scalar(
#         lambda x: -double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2),
#         bounds=bounds,
#         method='bounded'
#     )
    
#     if result.success:
#         return result.x, -result.fun  # Return position and value of maximum
#     else:
#         # Fallback to midpoint if optimization fails
#         x_max = (mu1 + mu2) / 2
#         return x_max, double_gaussian(x_max, A1, mu1, sigma1, A2, mu2, sigma2)


# def find_threshold_crossings(f, popt, max_val, rel_th=0.01, x_range=(-10, 10), tol=1e-2):
#     """
#     Finds the x values where the double Gaussian drops below rel_th * max_val.
    
#     Returns:
#         q_low: lower bound crossing point
#         q_high: upper bound crossing point
#     """
#     threshold = rel_th * max_val

#     # Define shifted function for root finding
#     def shifted_func(x):
#         return f(x, *popt) - threshold

#     # Sample the function to find intervals where the threshold is crossed
#     x_vals = np.linspace(*x_range, 1000)
#     y_vals = [shifted_func(x) for x in x_vals]

#     # Find sign changes: threshold crossings
#     crossings = []
#     for i in range(len(x_vals) - 1):
#         if y_vals[i] * y_vals[i + 1] < 0:  # Sign change implies root in interval
#             try:
#                 root = brentq(shifted_func, x_vals[i], x_vals[i + 1], xtol=tol)
#                 crossings.append(root)
#             except ValueError:
#                 continue

#     # Return first and last crossing as bounds
#     if len(crossings) >= 2:
#         return crossings[0], crossings[-1]
#     else:
#         return None, None  # or raise Exception("Could not determine bounds")

# t_sum_gaussian_rel_th = 0.05
# fit_results = {}  # store results if needed
# copied_working_df = working_df.copy()

# for plane in range(1, 5):
#     for strip in range(1, 5):
#         col = f"T{plane}_T_sum_{strip}"
#         series = working_df[col]
#         nonzero = series[series != 0]

#         if len(nonzero) < 100:
#             print(f"Skipping {col}: too few entries ({len(nonzero)})")
#             continue

#         # Histogram for fitting
#         hist_vals, bin_edges = np.histogram(nonzero, bins=200, density=True)
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#         # Initial parameter guesses
#         A1_guess = np.max(hist_vals)
#         mu1_guess = np.median(nonzero)
#         sigma1_guess = np.std(nonzero) / 2
#         A2_guess = A1_guess / 2
#         mu2_guess = mu1_guess
#         sigma2_guess = np.std(nonzero)

#         try:
#             popt, _ = curve_fit(
#                 double_gaussian,
#                 bin_centers,
#                 hist_vals,
#                 p0=[A1_guess, mu1_guess, sigma1_guess, A2_guess, mu2_guess, sigma2_guess],
#                 bounds=(
#                     [0, -5, 0.1, 0, -5, 0.1],  # Lower bounds
#                     [np.inf, 5, 20, np.inf, 5, 20]  # Upper bounds
#                 ),
#                 maxfev=5000
#             )
#         except RuntimeError:
#             print(f"Fit failed for {col}")
#             continue

#         # Save result
#         fit_results[col] = popt
#         A1, mu1, sigma1, A2, mu2, sigma2 = popt

#         # q_low, q_high = ( mu1 + mu2 ) / 2 - 3 * ( sigma1 + sigma2 ) / 2, ( mu1 + mu2 ) / 2 + 3 * ( sigma1 + sigma2 ) / 2
        
#         max_x, max_val = find_true_max(*popt)
#         q_low, q_high = find_threshold_crossings(double_gaussian, popt, max_val, rel_th=t_sum_gaussian_rel_th)

#         mask = (series >= q_low) & (series <= q_high)
#         working_df.loc[~mask & (series != 0), col] = 0.0


# if create_essential_plots or create_plots:
# # if create_plots:
#     fig, axs = plt.subplots(4, 4, figsize=(20, 16))
#     fig.suptitle("Double Gaussian Fits for $T_\\mathrm{sum}$ Distributions", fontsize=16)

#     for plane in range(1, 5):
#         for strip in range(1, 5):
#             col = f"T{plane}_T_sum_{strip}"
#             series = copied_working_df[col]
#             nonzero = series[series != 0]

#             ax = axs[plane - 1, strip - 1]

#             if len(nonzero) < 100 or col not in fit_results:
#                 ax.set_title(f"{col} (no fit)")
#                 ax.axis("off")
#                 continue

#             hist_vals, bin_edges = np.histogram(nonzero, bins=200, density=True)
#             bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#             popt = fit_results[col]
#             A1, mu1, sigma1, A2, mu2, sigma2 = popt
            
#             # q_low, q_high = ( mu1 + mu2 ) / 2 - 3 * ( sigma1 + sigma2 ) / 2, ( mu1 + mu2 ) / 2 + 3 * ( sigma1 + sigma2 ) / 2
        
#             # Find the maximum value of the fitted function
#             x_max, max_val = find_true_max(*popt)
#             print(f"True maximum at x = {x_max:.2f}, value = {max_val:.2f}")
            
#             max_x, max_val = find_true_max(*popt)
#             q_low, q_high = find_threshold_crossings(double_gaussian, popt, max_val, rel_th=t_sum_gaussian_rel_th)
            
#             x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
#             y_fit = double_gaussian(x_fit, *popt)
            
#             ax.axvline(q_low, color='red', linestyle='--', label='Lower Limit', alpha=0.7)
#             ax.axvline(q_high, color='red', linestyle='--', label='Upper Limit', alpha=0.7)
#             ax.plot(bin_centers, hist_vals, lw=1.5, label="Data", alpha=0.6)
#             ax.plot(x_fit, y_fit, lw=2.0, label="Fit")
#             ax.set_xlim(-3, 3)
#             ax.set_title(f"{col}")
#             ax.tick_params(labelsize=8)

#     plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#     if save_plots:
#         name_of_file = f'gaussian_timing_{plane}_{strip}'
#         final_filename = f'{fig_idx}_{name_of_file}.png'
#         fig_idx += 1
#         save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
#         plot_list.append(save_fig_path)
#         plt.savefig(save_fig_path, format='png')
#     if show_plots:
#         plt.show()
#     plt.close()


print("----------------------------------------------------------------------")
print("---------- Filter if any variable in the strip is 0 (4/3) ------------")
print("----------------------------------------------------------------------")

# Now go throuhg every plane and strip and if any of the T_sum, T_diff, Q_sum, Q_diff == 0,
# put the four variables in that plane, strip and event to 0

total_events = len(working_df)

for plane in range(1, 5):
    for strip in range(1, 5):
        q_sum  = f'Q{plane}_Q_sum_{strip}'
        q_diff = f'Q{plane}_Q_diff_{strip}'
        t_sum  = f'T{plane}_T_sum_{strip}'
        t_diff = f'T{plane}_T_diff_{strip}'
        
        # Build mask
        mask = (
            (working_df[q_sum]  == 0) |
            (working_df[q_diff] == 0) |
            (working_df[t_sum]  == 0) |
            (working_df[t_diff] == 0)
        )
        
        # Count affected events
        num_affected_events = mask.sum()
        print(f"Plane {plane}, Strip {strip}: {num_affected_events} out of {total_events} events affected ({(num_affected_events / total_events) * 100:.2f}%)")

        # Zero the affected values
        working_df.loc[mask, [q_sum, q_diff, t_sum, t_diff]] = 0


print("----------------------------------------------------------------------")
print("--------------------- Defining preprocessed_tt -----------------------")
print("----------------------------------------------------------------------")

def compute_preprocessed_tt(row):
    name = ''
    for plane in range(1, 5):
        this_plane = False
        for strip in range(1, 5):
            q_sum_col  = f'Q{plane}_Q_sum_{strip}'
            q_diff_col = f'Q{plane}_Q_diff_{strip}'
            t_sum_col  = f'T{plane}_T_sum_{strip}'
            t_diff_col = f'T{plane}_T_diff_{strip}'
            
            if (row[q_sum_col] != 0 and row[q_diff_col] != 0 and
                row[t_sum_col] != 0 and row[t_diff_col] != 0):
                this_plane = True
                break  # One valid strip is enough to consider the plane valid
        if this_plane:
            name += str(plane)
    return int(name) if name else 0  # Return 0 if no plane is valid

# Apply to all rows
working_df["preprocessed_tt"] = working_df.apply(compute_preprocessed_tt, axis=1)

if self_trigger:
    working_st_df["preprocessed_tt"] = working_st_df.apply(compute_preprocessed_tt, axis=1)


if time_window_filtering:    
    print("----------------------------------------------------------------------")
    print("-------------------- Time window filtering (3/3) ---------------------")
    print("----------------------------------------------------------------------")
    
    # Pre removal of outliers
    spread_results = []
    for preprocessed_tt in sorted(working_df["preprocessed_tt"].unique()):
        filtered_df = working_df[working_df["preprocessed_tt"] == preprocessed_tt].copy()
        T_sum_columns_tt = filtered_df.filter(regex='_T_sum_').columns
        t_sum_spread_tt = filtered_df[T_sum_columns_tt].apply(lambda row: np.ptp(row[row != 0]) if np.any(row != 0) else np.nan, axis=1)
        filtered_df["T_sum_spread"] = t_sum_spread_tt
        spread_results.append(filtered_df)
    spread_df = pd.concat(spread_results, ignore_index=True)

    # if create_plots:
    if create_essential_plots or create_plots:
        fig, axs = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=False)
        axs = axs.flatten()
        for i, tt in enumerate(sorted(spread_df["preprocessed_tt"].unique())):
            subset = spread_df[spread_df["preprocessed_tt"] == tt]
            v = subset["T_sum_spread"].dropna()
            v = v[v < coincidence_window_cal_ns * 2]
            axs[i].hist(v, bins=100, alpha=0.7)
            axs[i].set_title(f"TT = {tt}")
            axs[i].set_xlabel("ΔT (ns)")
            axs[i].set_ylabel("Events")
            axs[i].axvline(x=coincidence_window_cal_ns, color='red', linestyle='--', label='Time coincidence window')
            # Logscale
            axs[i].set_yscale('log')
        fig.suptitle("Non filtered. Intra-Event T_sum Spread by preprocessed_tt")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if save_plots:
            hist_filename = f'{fig_idx}_tsum_spread_histograms.png'
            fig_idx += 1
            hist_path = os.path.join(base_directories["figure_directory"], hist_filename)
            plot_list.append(hist_path)
            fig.savefig(hist_path, format='png')
        if show_plots: plt.show()
        plt.close(fig)

    # Removal of outliers
    def zero_outlier_tsum(row, threshold=coincidence_window_cal_ns):
        t_sum_cols = [col for col in row.index if '_T_sum_' in col]
        t_sum_vals = row[t_sum_cols].copy()
        nonzero_vals = t_sum_vals[t_sum_vals != 0]
        if len(nonzero_vals) < 2: return row
        center = np.mean(nonzero_vals)
        deviations = np.abs(nonzero_vals - center)
        outliers = deviations > threshold / 2
        for col in outliers.index[outliers]: row[col] = 0.0
        return row
    working_df = working_df.apply(zero_outlier_tsum, axis=1)

    # Post removal of outliers
    spread_results = []
    for preprocessed_tt in sorted(working_df["preprocessed_tt"].unique()):
        filtered_df = working_df[working_df["preprocessed_tt"] == preprocessed_tt].copy()
        T_sum_columns_tt = filtered_df.filter(regex='_T_sum_').columns
        t_sum_spread_tt = filtered_df[T_sum_columns_tt].apply(lambda row: np.ptp(row[row != 0]) if np.any(row != 0) else np.nan, axis=1)
        filtered_df["T_sum_spread"] = t_sum_spread_tt
        spread_results.append(filtered_df)
    spread_df = pd.concat(spread_results, ignore_index=True)

    # if create_plots:
    if create_essential_plots or create_plots:
        fig, axs = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=False)
        axs = axs.flatten()
        for i, tt in enumerate(sorted(spread_df["preprocessed_tt"].unique())):
            subset = spread_df[spread_df["preprocessed_tt"] == tt]
            v = subset["T_sum_spread"].dropna()
            axs[i].hist(v, bins=100, alpha=0.7)
            axs[i].set_title(f"TT = {tt}")
            axs[i].set_xlabel("ΔT (ns)")
            axs[i].set_ylabel("Events")
            axs[i].axvline(x=coincidence_window_cal_ns, color='red', linestyle='--', label='Time coincidence window')# Logscale
            axs[i].set_yscale('log')
        fig.suptitle("Cleaned. Corrected Intra-Event T_sum Spread by preprocessed_tt")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        if save_plots:
            hist_filename = f'{fig_idx}_tsum_spread_histograms_filtered.png'
            fig_idx += 1
            hist_path = os.path.join(base_directories["figure_directory"], hist_filename)
            plot_list.append(hist_path)
            fig.savefig(hist_path, format='png')
        if show_plots: plt.show()
        plt.close(fig)


    if create_plots:
    # if create_essential_plots or create_plots:
        # Identify all _T_sum_ columns
        T_sum_columns = working_df.filter(regex='_T_sum_').columns
        replaced_count = 0  # Global counter

        for preprocessed_tt in [  12 ,  23,   34 ,1234 , 123 , 234,  124  , 13  , 14 ,24 , 134]:
            mask = working_df['preprocessed_tt'] == preprocessed_tt
            filtered_df = working_df[mask].copy()  # Work on a copy for fitting

            if len(filtered_df) == 0:
                continue

            # Extract filtered T_sum data
            t_sum_data = filtered_df[T_sum_columns].values
            if not np.any(t_sum_data != 0):
                print(f"[Warning] Skipping Preprocessed TT {preprocessed_tt}: all T_sum values filtered out.")
                continue

            widths = np.linspace(0, coincidence_window_cal_ns, coincidence_window_cal_number_of_points)
            counts_per_width = []
            counts_per_width_dev = []

            for w in widths:
                count_in_window = []
                for row in t_sum_data:
                    row_no_zeros = row[row != 0]
                    if len(row_no_zeros) == 0:
                        count_in_window.append(0)
                        continue
                    stat = np.median(row_no_zeros)  # Or mean
                    lower = stat - w / 2
                    upper = stat + w / 2
                    n_in_window = np.sum((row_no_zeros >= lower) & (row_no_zeros <= upper))
                    count_in_window.append(n_in_window)
                counts_per_width.append(np.mean(count_in_window))
                counts_per_width_dev.append(np.std(count_in_window))

            counts_per_width = np.array(counts_per_width)
            counts_per_width_dev = np.array(counts_per_width_dev)
            valid_mask = np.isfinite(counts_per_width) & (counts_per_width > 0)
            if not np.any(valid_mask):
                print(f"[Warning] Skipping Preprocessed TT {preprocessed_tt}: no valid window accumulation.")
                continue
            counts_per_width_norm = counts_per_width / np.max(counts_per_width)
            # counts_per_width_norm = counts_per_width

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(widths, counts_per_width_norm, label='Normalized average count in window', color='blue', s=30)
            ax.axvline(x=coincidence_window_cal_ns, color='red', linestyle='--', label='Time coincidence window')
            ax.set_xlabel("Window width (ns)")
            ax.set_ylabel("Normalized average # of T_sum values in window")
            ax.set_title(f"Fraction of hits within stat-centered window vs width (TT = {preprocessed_tt})")
            ax.grid(True)
            ax.legend()

            if save_plots:
                name_of_file = f'stat_window_accumulation_{preprocessed_tt}'
                final_filename = f'{fig_idx}_{name_of_file}.png'
                fig_idx += 1
                save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                plot_list.append(save_fig_path)
                plt.savefig(save_fig_path, format='png')
            if show_plots:
                plt.show()
            plt.close()


print("----------------------------------------------------------------------")
print("---------- Filter if any variable in the strip is 0 (5/3) ------------")
print("----------------------------------------------------------------------")

# Now go throuhg every plane and strip and if any of the T_sum, T_diff, Q_sum, Q_diff == 0,
# put the four variables in that plane, strip and event to 0

total_events = len(working_df)

for plane in range(1, 5):
    for strip in range(1, 5):
        q_sum  = f'Q{plane}_Q_sum_{strip}'
        q_diff = f'Q{plane}_Q_diff_{strip}'
        t_sum  = f'T{plane}_T_sum_{strip}'
        t_diff = f'T{plane}_T_diff_{strip}'
        
        # Build mask
        mask = (
            (working_df[q_sum]  == 0) |
            (working_df[q_diff] == 0) |
            (working_df[t_sum]  == 0) |
            (working_df[t_diff] == 0)
        )
        
        # Count affected events
        num_affected_events = mask.sum()
        print(f"Plane {plane}, Strip {strip}: {num_affected_events} out of {total_events} events affected ({(num_affected_events / total_events) * 100:.2f}%)")

        # Zero the affected values
        working_df.loc[mask, [q_sum, q_diff, t_sum, t_diff]] = 0

working_df = working_df.copy()

print("----------------------------------------------------------------------")
print("--------------------- Defining posfiltered_tt -----------------------")
print("----------------------------------------------------------------------")

def compute_posfiltered_tt(row):
    name = ''
    for plane in range(1, 5):
        this_plane = False
        for strip in range(1, 5):
            q_sum_col  = f'Q{plane}_Q_sum_{strip}'
            q_diff_col = f'Q{plane}_Q_diff_{strip}'
            t_sum_col  = f'T{plane}_T_sum_{strip}'
            t_diff_col = f'T{plane}_T_diff_{strip}'
            
            if (row[q_sum_col] != 0 and row[q_diff_col] != 0 and
                row[t_sum_col] != 0 and row[t_diff_col] != 0):
                this_plane = True
                break  # One valid strip is enough to consider the plane valid
        if this_plane:
            name += str(plane)
    return int(name) if name else 0  # Return 0 if no plane is valid

# Apply to all rows
working_df["posfiltered_tt"] = working_df.apply(compute_posfiltered_tt, axis=1)
working_df = working_df.copy()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Save the metadata, calibrations and monitoring stuff ------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


the_filename = "placeholder"
analysis_date = datetime.now().strftime("%Y-%m-%d")
print(f"Analysis date and time: {analysis_date}")



# Construct the new calibration row

# Current time of the analysis
new_row = {'Filename': the_filename, 'Analysis_Date': analysis_date, 'Start_Time': start_time, 'End_Time': end_time}

# Include pedestal and calibration parameters
for i, module in enumerate(['P1', 'P2', 'P3', 'P4']):
    for j in range(4):
        strip = j + 1
        if crosstalk_fitting:
            q_sum = (QF_pedestal[i][j] + QB_pedestal[i][j]) / 2 - crosstalk_pedestal[f'crstlk_pedestal_{module}s{strip}']
        else:
            q_sum = (QF_pedestal[i][j] + QB_pedestal[i][j]) / 2
        new_row[f'{module}_s{strip}_Q_sum'] = q_sum
        new_row[f'{module}_s{strip}_Q_F'] = QF_pedestal[i][j]
        new_row[f'{module}_s{strip}_Q_B'] = QB_pedestal[i][j]
        new_row[f'{module}_s{strip}_T_sum'] = calibration_times[i, j]
        new_row[f'{module}_s{strip}_T_dif'] = Tdiff_cal[i][j]

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
    print(f"Updated existing metadata for time range: {start_time} to {end_time}")
else:
    metadata_df = pd.concat([metadata_df, pd.DataFrame([new_row])], ignore_index=True)
    print(f"Added new metadata for time range: {start_time} to {end_time}")

# Sort and save
metadata_df.sort_values(by='Start_Time', inplace=True)

# Put Start_Time and End_Time as first columns
metadata_df = metadata_df[['Filename', 'Analysis_Date', 'Start_Time', 'End_Time'] + [col for col in metadata_df.columns if col not in ['Start_Time', 'End_Time']]]

metadata_df.to_csv(csv_path, index=False, float_format='%.5g')
print(f'{csv_path} updated with the calibration summary.')



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
                    pdf_save_rasterized_page(pdf, fig, bbox_inches='tight')  # Save figure tightly fitting the image
                    plt.close(fig)  # Close the figure after adding it to the PDF

        # Remove PNG files after creating the PDF
        for png in plot_list:
            try:
                os.remove(png)
                # print(f"Deleted {png}")
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")



# Path to save the cleaned dataframe
# Create output directory if it does not exist /home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_1/DONE/
os.makedirs(f"{output_directory}", exist_ok=True)
OUT_PATH = f"{output_directory}/calibrated_{basename_no_ext}.h5"
KEY = "df"  # HDF5 key name

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# --- Example: your cleaned DataFrame is called working_df ---
# (Here, you would have your data cleaning code before saving)
# working_df = ...



# Print all column names in the dataframe
print("Columns in the cleaned dataframe:")
for col in working_df.columns:
    print(col)









def _collect_columns(columns: Iterable[str], pattern: re.Pattern[str]) -> list[str]:
    """Return all column names that match *pattern*."""
    return [name for name in columns if pattern.match(name)]


# Pattern for Q4_Q_sum_3, Q4_Q_sum_4, Q3_Q_sum_3...
Q_SUM_PATTERN = re.compile(r"^Q\d+_Q_sum_\d+$")


# If Q*_F_* and Q*_B_* are zero for all cases, remove the row
Q_cols = _collect_columns(working_df.columns, Q_SUM_PATTERN)
working_df = working_df[(working_df[Q_cols] != 0).any(axis=1)]






print(f"Original number of events in the dataframe: {original_number_of_events}")
# Final number of events
final_number_of_events = len(working_df)
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

print("----------\nMetadata to be saved:")
print(f"Filename base: {filename_base}")
print(f"Execution timestamp: {execution_timestamp}")
print(f"Data purity percentage: {data_purity_percentage:.2f}%")
print(f"Total execution time: {total_execution_time_minutes:.2f} minutes\n----------")

metadata_csv_path = save_execution_metadata(
    home_path,
    station,
    task_number,
    {
        "filename_base": filename_base,
        "execution_timestamp": execution_timestamp,
        "data_purity_percentage": round(float(data_purity_percentage), 4),
        "total_execution_time_minutes": round(float(total_execution_time_minutes), 4),
    },
)
print(f"Metadata CSV updated at: {metadata_csv_path}")


# Save to HDF5 file
working_df.to_hdf(OUT_PATH, key=KEY, mode="w", format="table")
print(f"Calibrated dataframe saved to: {OUT_PATH}")


# Move the original datafile to COMPLETED -------------------------------------
print("Moving file to COMPLETED directory...")

if user_file_selection == False:
    shutil.move(file_path, completed_file_path)
    now = time.time()
    os.utime(completed_file_path, (now, now))
    print("************************************************************")
    print(f"File moved from\n{file_path}\nto:\n{completed_file_path}")
    print("************************************************************")
