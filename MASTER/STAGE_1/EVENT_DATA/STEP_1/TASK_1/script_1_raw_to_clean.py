#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%

from __future__ import annotations

"""
Created on Thu Jun 20 09:15:33 2024

@author: csoneira@ucm.es
"""


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# --------------- TASK_1: import and clean ------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


task_number = 1



print("----------------------------------------------------------------------")
print("-------------------- STAGE_0_to_1 TO LIST CLEAN IS STARTING -------------------")
print("----------------------------------------------------------------------")


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



from datetime import datetime, timedelta

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

import yaml

# Warning Filters
warnings.filterwarnings("ignore", message=".*Data has no positive values, and therefore cannot be log-scaled.*")



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


# -----------------------------------------------------------------------------

# Round execution time to seconds and format it in YYYY-MM-DD_HH.MM.SS
execution_time = str(start_execution_time_counting).split('.')[0]  # Remove microseconds
print("Execution time is:", execution_time)

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

# -----------------------------------------------------------------------------

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
    raw_working_directory = os.path.join(station_directory, raw_directory)
    
else:
    raw_directory = f"STEP_1/TASK_{task_number - 1}/OUTPUT_FILES"
    raw_working_directory = os.path.join(base_directory, raw_directory)

if task_number == 5:
    output_location = os.path.join(base_directory, "STEP_1_TO_2_OUTPUT")
else:
    output_location = os.path.join(raw_to_list_working_directory, "OUTPUT_FILES")


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

# Move files from STAGE_0_to_1 to STAGE_0_to_1_TO_LIST/FILES/UNPROCESSED,
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

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Header ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

user_home = os.path.expanduser("~")
config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config_global.yaml")
parameter_config_file_path = os.path.join(user_home, "DATAFLOW_v3/MASTER/CONFIG_FILES/config_parameters.csv")
print(f"Using config file: {config_file_path}")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
home_path = config["home_path"]
config = update_config_with_parameters(config, parameter_config_file_path, station)

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


# the analysis mode indicates if it is a regular analysis or a repeated, careful analysis
# 0 -> regular analysis
# 1 -> repeated, careful analysis
global_variables = {
    'analysis_mode': 0,
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
print(f"File to be processed, complete original path: {file_path}")
the_filename = os.path.basename(file_path)
print(f"File to process: {the_filename}")
basename_no_ext, file_extension = os.path.splitext(the_filename)
print(f"File basename (no extension): {basename_no_ext}")

analysis_date = datetime.now().strftime("%Y-%m-%d")
# print(f"Analysis date and time: {analysis_date}")

# Modify the time of the processing file to the current time so it looks fresh
now = time.time()
os.utime(processing_file_path, (now, now))

# Check the station number in the datafile
# It might be that the data header is, instead of mi01: minI, which is the same, in that
# case consider minI as mi01
try:
    station_label = file_name[3]  # 4th character (index 3)
    print(f'File station number is: {station_label}')
    
    if station_label == "I":
        print("Station label is 'I', interpreting as station 1.")
        station_label = int(1)

    file_station_number = int(station_label)  # 4th character (index 3)
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

# ------------------------------------------------------------------------------------------------------

# Move rejected_file to the rejected file folder
temp_file = os.path.join(base_directories["temp_files_directory"], f"temp_file_{date_execution}.csv")
rejected_file = os.path.join(base_directories["rejected_files_directory"], f"temp_file_{date_execution}.csv")

print(f"Temporal file is {temp_file}")
EXPECTED_COLUMNS = EXPECTED_COLUMNS_config  # Expected number of columns

ZERO_TOKEN_PATTERN = re.compile(r"0000\.0000")
LEADING_ZERO_PATTERN = re.compile(r"\b0+([0-9]+)")
MULTI_SPACE_PATTERN = re.compile(r" +")
XYEAR_PATTERN = re.compile(r"X(20\d{2})")
NEG_GAP_PATTERN = re.compile(r"(\w)-(\d)")
MALFORMED_NUMBER_PATTERN = re.compile(r"-?\d+\.\d+\.\d+")
VALID_YEARS = set(range(2022, 2033))

T_FRONT_PATTERN = re.compile(r"^T\d+_F_\d+$")
T_BACK_PATTERN = re.compile(r"^T\d+_B_\d+$")
Q_FRONT_PATTERN = re.compile(r"^Q\d+_F_\d+$")
Q_BACK_PATTERN = re.compile(r"^Q\d+_B_\d+$")

def _apply_bounds(frame: pd.DataFrame, column_names: Iterable[str], lower: float, upper: float) -> None:
    """Zero out values outside [lower, upper] for the provided columns."""
    cols = tuple(column_names)
    if not cols:
        return
    subset = frame.loc[:, cols]
    frame.loc[:, cols] = subset.where((subset >= lower) & (subset <= upper), 0)

def _collect_columns(columns: Iterable[str], pattern: re.Pattern[str]) -> list[str]:
    """Return all column names that match *pattern*."""
    return [name for name in columns if pattern.match(name)]

# Function to process each line
def process_line(line):
    line = ZERO_TOKEN_PATTERN.sub('0', line)  # Replace '0000.0000' with '0'
    line = LEADING_ZERO_PATTERN.sub(r'\1', line)  # Remove leading zeros
    line = MULTI_SPACE_PATTERN.sub(',', line.strip())  # Replace multiple spaces with a comma
    line = XYEAR_PATTERN.sub(r'X\n\1', line)  # Replace XYYYY with X\nYYYY
    line = NEG_GAP_PATTERN.sub(r'\1 -\2', line)  # Ensure X-Y is properly spaced
    return line

# Function to check for malformed numbers (e.g., '-120.144.0')
def contains_malformed_numbers(line):
    return bool(MALFORMED_NUMBER_PATTERN.search(line))  # Detects multiple decimal points

# Function to validate year, month, and day
def is_valid_date(values):
    try:
        year, month, day = int(values[0]), int(values[1]), int(values[2])
        if year not in VALID_YEARS:  # Check valid years
            return False
        if not (1 <= month <= 12):  # Check valid month
            return False
        if not (1 <= day <= 31):  # Check valid day
            return False
        return True
    except ValueError:  # In case of non-numeric values
        return False









# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------- 
# -----------------------------------------------------------------------------------------------------------
# TASK 1 start
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

# Process the file
read_lines = 0
written_lines = 0
with open(file_path, 'r') as infile, open(temp_file, 'w') as outfile, open(rejected_file, 'w') as rejectfile:
    for i, line in enumerate(infile, start=1):
        read_lines += 1
        
        cleaned_line = process_line(line)
        cleaned_values = cleaned_line.split(',')  # Split into columns

        # Validate line structure before further processing
        if len(cleaned_values) < 3 or not is_valid_date(cleaned_values[:3]):
            rejectfile.write(f"Line {i} (Invalid date): {line.strip()}\n")
            continue  # Skip this row

        if contains_malformed_numbers(line):
            rejectfile.write(f"Line {i} (Malformed number): {line.strip()}\n")  # Save rejected row
            continue  # Skip this row

        # Ensure correct column count
        if len(cleaned_values) == EXPECTED_COLUMNS:
            written_lines += 1
            outfile.write(cleaned_line + '\n')  # Save valid row
        else:
            rejectfile.write(f"Line {i} (Wrong column count): {line.strip()}\n")  # Save rejected row

read_df = pd.read_csv(
    temp_file,
    header=None,
    engine="c",
    dtype=np.float64,
    na_values=["", " "],
    keep_default_na=True,
    nrows=limit_number if limit else None,
)

# Print the number of rows in input
print(f"\nOriginal file has {read_lines} lines.")
print(f"Processed file has {written_lines} lines.")
valid_lines_in_dat_file = written_lines/read_lines * 100
print(f"--> A {valid_lines_in_dat_file:.2f}% of the lines were valid.\n")

global_variables['valid_lines_in_binary_file_percentage'] =  valid_lines_in_dat_file

# Assign name to the columns
read_df.columns = ['year', 'month', 'day', 'hour', 'minute', 'second'] + [f'column_{i}' for i in range(6, 71)]
time_columns = ['year', 'month', 'day', 'hour', 'minute', 'second']
for col in time_columns:
    read_df[col] = read_df[col].round().astype("Int64")

read_df['datetime'] = pd.to_datetime(read_df[time_columns])
value_columns = [col for col in read_df.columns if col not in (*time_columns, 'datetime')]
if 'column_6' in read_df.columns:
    read_df['column_6'] = read_df['column_6'].round().astype("Int64")
    value_columns = [col for col in value_columns if col != 'column_6']
if value_columns:
    read_df[value_columns] = read_df[value_columns].astype(np.float32, copy=False)


print("----------------------------------------------------------------------")
print("-------------------------- Filter 1: by date -------------------------")
print("----------------------------------------------------------------------")

selected_df = read_df.loc[read_df['datetime'].between(left_limit_time, right_limit_time)].copy()
del read_df
gc.collect()
if not isinstance(selected_df.set_index('datetime').index, pd.DatetimeIndex):
    raise ValueError("The index is not a DatetimeIndex. Check 'datetime' column formatting.")

# Print the count frequency of the values in column_6
print(selected_df['column_6'].value_counts())
# Take only the rows in which column_6 is equal to 1

self_trigger_df = selected_df[selected_df['column_6'] == 2]
selected_df = selected_df[selected_df['column_6'] == 1]
self_trigger = not self_trigger_df.empty # If self_trigger_df has values, define an indicator as True

raw_data_len = len(selected_df)
if raw_data_len == 0 and not self_trigger:
    print("No coincidence nor self-trigger events.")
    sys.exit(1)

# Note that the middle between start and end time could also be taken. This is for calibration storage.
datetime_value = selected_df['datetime'].iloc[0]
end_datetime_value = selected_df['datetime'].iloc[-1]

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
        save_pdf_filename = "essential_" + save_pdf_filename

save_pdf_path = os.path.join(base_directories["pdf_directory"], save_pdf_filename)

# Check if the file exists and its size
if os.path.exists(save_filename):
    if os.path.getsize(save_filename) >= 1 * 1024 * 1024: # Bigger than 1MB
        if force_replacement == False:
            print("Datafile found and it looks completed. Exiting...")
            sys.exit()  # Exit the script
        else:
            print("Datafile found and it is not empty, but 'force_replacement' is True, so it creates new datafiles anyway.")
    else:
        print("Datafile found, but empty.")

column_indices = {
    'T1_F': range(55, 59), 'T1_B': range(59, 63), 'Q1_F': range(63, 67), 'Q1_B': range(67, 71),
    'T2_F': range(39, 43), 'T2_B': range(43, 47), 'Q2_F': range(47, 51), 'Q2_B': range(51, 55),
    'T3_F': range(23, 27), 'T3_B': range(27, 31), 'Q3_F': range(31, 35), 'Q3_B': range(35, 39),
    'T4_F': range(7, 11), 'T4_B': range(11, 15), 'Q4_F': range(15, 19), 'Q4_B': range(19, 23)
}

# Extract and assign appropriate column names
columns_data = {'datetime': selected_df['datetime'].values}
for key, idx_range in column_indices.items():
    for i, col_idx in enumerate(idx_range):
        column_name = f'{key}_{i+1}'
        columns_data[column_name] = selected_df.iloc[:, col_idx].values

# Create a DataFrame from the columns data
working_df = pd.DataFrame(columns_data)
original_number_of_events = len(working_df)

working_df["datetime"] = selected_df['datetime']

if found_matching_conf:
    # --- Conditional swap for station 2, Plane 4: swap channels 2 and 4 ---
    if selected_conf['conf'] < 2:
        if station == "2":
            print("Configuration of the detector is less than 2.")
            print("Swapping channels that give problems in plane 4.")
            plane4_keys = ['T4_F', 'T4_B', 'Q4_F', 'Q4_B']
            for key in plane4_keys:
                col2 = f'{key}_3'
                col4 = f'{key}_4'
                working_df[[col2, col4]] = working_df[[col4, col2]].values  # swap columns

if self_trigger:
    # Extract and assign appropriate column names
    columns_data = {'datetime': self_trigger_df['datetime'].values}
    for key, idx_range in column_indices.items():
        for i, col_idx in enumerate(idx_range):
            column_name = f'{key}_{i+1}'
            columns_data[column_name] = self_trigger_df.iloc[:, col_idx].values

    # Create a DataFrame from the columns data
    working_st_df = pd.DataFrame(columns_data)
    working_st_df["datetime"] = self_trigger_df['datetime']
    
    if found_matching_conf:
        # --- Conditional swap for station 2, Plane 4: swap channels 2 and 4 ---
        if selected_conf['conf'] < 2:
            if station == "2":
                print("Configuration of the detector is less than 2.")
                print("Swapping channels that give problems in plane 4.")
                plane4_keys = ['T4_F', 'T4_B', 'Q4_F', 'Q4_B']
                for key in plane4_keys:
                    col2 = f'{key}_3'
                    col4 = f'{key}_4'
                    working_st_df[[col2, col4]] = working_st_df[[col4, col2]].values  # swap columns


# ----------------------------------------------------------------------------------
# Count the number of non-zero entries per channel in the whole dataframe ----------
# ----------------------------------------------------------------------------------

# Count per each column the number of non-zero entries and save it in a column of
# global_variables called TX_F_Y_entries or TX_B_Y_entries

# Count for main dataframe (non-self-trigger)
for key, idx_range in column_indices.items():
    for i in range(1, len(idx_range) + 1):
        colname = f"{key}_{i}"
        count = (working_df[colname] != 0).sum()
        global_var_name = f"{key}_{i}_entries_original"
        global_variables[global_var_name] = count


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# Original trigger type ------------------------------------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------

# Now obtain the trigger type
def create_original_tt(df):
    def get_original_tt(row):
        planes_with_charge = []
        for plane in range(1, 5):
            charge_columns = [f'Q{plane}_F_1', f'Q{plane}_F_2', f'Q{plane}_F_3', f'Q{plane}_F_4',
                              f'Q{plane}_B_1', f'Q{plane}_B_2', f'Q{plane}_B_3', f'Q{plane}_B_4']
            if any(row[col] != 0 for col in charge_columns):
                planes_with_charge.append(str(plane))
        return ''.join(planes_with_charge)
    
    df['original_tt'] = df.apply(get_original_tt, axis=1)
    return df

# Apply the function to the DataFrame
working_df = create_original_tt(working_df)
working_df['original_tt'] = working_df['original_tt'].apply(builtins.int)

if self_trigger:
    working_st_df = create_original_tt(working_st_df)
    working_st_df['original_tt'] = working_st_df['original_tt'].apply(builtins.int)

if create_plots :
    event_counts = working_df['original_tt'].value_counts()

    plt.figure(figsize=(10, 6))
    event_counts.plot(kind='bar', alpha=0.7)
    plt.title(f'Number of Events per Original TT Label, {start_time}')
    plt.xlabel('Original TT Label')
    plt.ylabel('Number of Events')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_plots:
        final_filename = f'{fig_idx}_original_TT.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close()
    

if self_trigger:
    if create_essential_plots or create_plots:
   
        event_counts = working_st_df['original_tt'].value_counts()

        plt.figure(figsize=(10, 6))
        event_counts.plot(kind='bar', alpha=0.7)
        plt.title(f'Number of Events per Original TT Label, {start_time}')
        plt.xlabel('Original TT Label')
        plt.ylabel('Number of Events')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            final_filename = f'{fig_idx}_original_TT_ST.png'
            fig_idx += 1

            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')

        if show_plots: plt.show()
        plt.close()


# -----------------------------------------------------------------------------
# New channel-wise plot -------------------------------------------------------
# -----------------------------------------------------------------------------

if create_plots:
    # Create the grand figure for T values
    fig_T, axes_T = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_T = axes_T.flatten()
    
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key}_F_{j+1}'
            col_B = f'{key}_B_{j+1}'
            y_F = working_df[col_F]
            y_B = working_df[col_B]
            
            # Plot histograms with T-specific clipping and bins
            axes_T[i*4 + j].hist(y_F[(y_F != 0) & (y_F > T_clip_min) & (y_F < T_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
            axes_T[i*4 + j].hist(y_B[(y_B != 0) & (y_B > T_clip_min) & (y_B < T_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
            axes_T[i*4 + j].axvline(x=T_F_left_pre_cal, color='red', linestyle='--', label='T_left_pre_cal')
            axes_T[i*4 + j].axvline(x=T_F_right_pre_cal, color='blue', linestyle='--', label='T_right_pre_cal')
            axes_T[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_T[i*4 + j].legend()
            
            if log_scale:
                axes_T[i*4 + j].set_yscale('log')  # For T values

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Grand Figure for T values, mingo0{station}\n{start_time}", fontsize=16)
    
    if save_plots:
        final_filename = f'{fig_idx}_grand_figure_T.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close(fig_T)

    # Create the grand figure for Q values
    fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_Q = axes_Q.flatten()
    
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key.replace("T", "Q")}_F_{j+1}'
            col_B = f'{key.replace("T", "Q")}_B_{j+1}'
            y_F = working_df[col_F]
            y_B = working_df[col_B]
            
            # Plot histograms with Q-specific clipping and bins
            axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
            axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min) & (y_B < Q_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
            axes_Q[i*4 + j].axvline(x=Q_F_left_pre_cal, color='red', linestyle='--', label='Q_left_pre_cal')
            axes_Q[i*4 + j].axvline(x=Q_F_right_pre_cal, color='blue', linestyle='--', label='Q_right_pre_cal')
            axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_Q[i*4 + j].legend()
            
            if log_scale:
                axes_Q[i*4 + j].set_yscale('log')  # For Q values

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Grand Figure for Q values, mingo0{station}\n{start_time}", fontsize=16)
    
    if save_plots:
        final_filename = f'{fig_idx}_grand_figure_Q.png'
        fig_idx += 1
        
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close(fig_Q)


if self_trigger:
    if create_plots or create_essential_plots:
   
        # Create the grand figure for T values
        fig_T, axes_T = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_T = axes_T.flatten()
        
        for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
            for j in range(4):
                col_F = f'{key}_F_{j+1}'
                col_B = f'{key}_B_{j+1}'
                y_F = working_st_df[col_F]
                y_B = working_st_df[col_B]
                
                # Plot histograms with T-specific clipping and bins
                axes_T[i*4 + j].hist(y_F[(y_F != 0) & (y_F > T_clip_min_ST) & (y_F < T_clip_max_ST)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
                axes_T[i*4 + j].hist(y_B[(y_B != 0) & (y_B > T_clip_min_ST) & (y_B < T_clip_max_ST)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
                axes_T[i*4 + j].axvline(x=T_F_left_pre_cal_ST, color='red', linestyle='--', label='T_left_pre_cal_ST')
                axes_T[i*4 + j].axvline(x=T_F_right_pre_cal_ST, color='blue', linestyle='--', label='T_right_pre_cal_ST')
                axes_T[i*4 + j].set_title(f'{col_F} vs {col_B}')
                axes_T[i*4 + j].legend()
                
                if log_scale:
                    axes_T[i*4 + j].set_yscale('log')  # For T values

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"SELF TRIGGER. Grand Figure for T values, mingo0{station}\n{start_time}", fontsize=16)
        
        if save_plots:
            final_filename = f'{fig_idx}_grand_figure_T_ST.png'
            fig_idx += 1

            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')

        if show_plots: plt.show()
        plt.close(fig_T)

        # Create the grand figure for Q values
        fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_Q = axes_Q.flatten()
        
        for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
            for j in range(4):
                col_F = f'{key.replace("T", "Q")}_F_{j+1}'
                col_B = f'{key.replace("T", "Q")}_B_{j+1}'
                y_F = working_st_df[col_F]
                y_B = working_st_df[col_B]
                
                # Plot histograms with Q-specific clipping and bins
                axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min_ST) & (y_F < Q_clip_max_ST)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
                axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min_ST) & (y_B < Q_clip_max_ST)], 
                                    bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
                axes_Q[i*4 + j].axvline(x=Q_F_left_pre_cal_ST, color='red', linestyle='--', label='Q_left_pre_cal_ST')
                axes_Q[i*4 + j].axvline(x=Q_F_right_pre_cal_ST, color='blue', linestyle='--', label='Q_right_pre_cal_ST')
                axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
                axes_Q[i*4 + j].legend()
                
                if log_scale:
                    axes_Q[i*4 + j].set_yscale('log')  # For Q values

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"SELF TRIGGER. Grand Figure for Q values, mingo0{station}\n{start_time}", fontsize=16)
        if save_plots:
            final_filename = f'{fig_idx}_grand_figure_Q_ST.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close(fig_Q)

# -----------------------------------------------------------------------------------------------

if create_plots:
    # Initialize figure and axes for scatter plot of Time vs Charge
    fig_TQ, axes_TQ = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_TQ = axes_TQ.flatten()

    # Iterate over each module (T1, T2, T3, T4)
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key}_F_{j+1}'  # Time F column
            col_B = f'{key}_B_{j+1}'  # Time B column
            
            y_F = working_df[col_F]  # Time values for front
            y_B = working_df[col_B]  # Time values for back
            
            charge_col_F = f'{key.replace("T", "Q")}_F_{j+1}'  # Corresponding charge column for front
            charge_col_B = f'{key.replace("T", "Q")}_B_{j+1}'  # Corresponding charge column for back
            
            charge_F = working_df[charge_col_F]  # Charge values for front
            charge_B = working_df[charge_col_B]  # Charge values for back
            
            # Apply clipping ranges to the data
            mask_F = (y_F != 0) & (y_F > T_clip_min) & (y_F < T_clip_max) & (charge_F > Q_clip_min) & (charge_F < Q_clip_max)
            mask_B = (y_B != 0) & (y_B > T_clip_min) & (y_B < T_clip_max) & (charge_B > Q_clip_min) & (charge_B < Q_clip_max)
            
            # Plot scatter plots for Time F vs Charge F and Time B vs Charge B
            axes_TQ[i*4 + j].scatter(charge_F[mask_F], y_F[mask_F], alpha=0.5, label=f'{col_F} (F)', color='green', s=1)
            axes_TQ[i*4 + j].scatter(charge_B[mask_B], y_B[mask_B], alpha=0.5, label=f'{col_B} (B)', color='orange', s=1)
            
            # Plot threshold lines for time and charge
            axes_TQ[i*4 + j].axhline(y=T_F_left_pre_cal, color='red', linestyle='--', label='T_left_pre_cal')
            axes_TQ[i*4 + j].axhline(y=T_F_right_pre_cal, color='blue', linestyle='--', label='T_right_pre_cal')
            axes_TQ[i*4 + j].axvline(x=Q_F_left_pre_cal, color='red', linestyle='--', label='Q_left_pre_cal')
            axes_TQ[i*4 + j].axvline(x=Q_F_right_pre_cal, color='blue', linestyle='--', label='Q_right_pre_cal')
            
            axes_TQ[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_TQ[i*4 + j].legend()

    # Adjust the layout and title
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Scatter Plot for T vs Q values, mingo0{station}\n{start_time}", fontsize=16)

    # Save the plot
    if save_plots:
        final_filename = f'{fig_idx}_scatter_plot_TQ.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    # Show the plot if requested
    if show_plots:
        plt.show()

    # Close the plot to avoid excessive memory usage
    plt.close(fig_TQ)

# -----------------------------------------------------------------------------

print("----------------------------------------------------------------------")
print("------------------ Filter 1.1.1: uncalibrated data -------------------")
print("----------------------------------------------------------------------")

# FILTER 2: TF, TB, QF, QB PRECALIBRATED THRESHOLDS --> 0 if out --------------

working_df.fillna(0, inplace=True)
T_F_cols = _collect_columns(working_df.columns, T_FRONT_PATTERN)
T_B_cols = _collect_columns(working_df.columns, T_BACK_PATTERN)
Q_F_cols = _collect_columns(working_df.columns, Q_FRONT_PATTERN)
Q_B_cols = _collect_columns(working_df.columns, Q_BACK_PATTERN)

_apply_bounds(working_df, T_F_cols, T_F_left_pre_cal, T_F_right_pre_cal)
_apply_bounds(working_df, T_B_cols, T_B_left_pre_cal, T_B_right_pre_cal)
_apply_bounds(working_df, Q_F_cols, Q_F_left_pre_cal, Q_F_right_pre_cal)
_apply_bounds(working_df, Q_B_cols, Q_B_left_pre_cal, Q_B_right_pre_cal)


if self_trigger:
    working_st_df.fillna(0, inplace=True)

    st_T_F_cols = _collect_columns(working_st_df.columns, T_FRONT_PATTERN)
    st_T_B_cols = _collect_columns(working_st_df.columns, T_BACK_PATTERN)
    st_Q_F_cols = _collect_columns(working_st_df.columns, Q_FRONT_PATTERN)
    st_Q_B_cols = _collect_columns(working_st_df.columns, Q_BACK_PATTERN)

    _apply_bounds(working_st_df, st_T_F_cols, T_F_left_pre_cal_ST, T_F_right_pre_cal_ST)
    _apply_bounds(working_st_df, st_T_B_cols, T_B_left_pre_cal_ST, T_B_right_pre_cal_ST)
    _apply_bounds(working_st_df, st_Q_F_cols, Q_F_left_pre_cal_ST, Q_F_right_pre_cal_ST)
    _apply_bounds(working_st_df, st_Q_B_cols, Q_B_left_pre_cal_ST, Q_B_right_pre_cal_ST)


# -----------------------------------------------------------------------------
# New channel-wise plot -------------------------------------------------------
# -----------------------------------------------------------------------------

if create_plots:
    # Create the grand figure for T values
    fig_T, axes_T = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_T = axes_T.flatten()
    
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key}_F_{j+1}'
            col_B = f'{key}_B_{j+1}'
            y_F = working_df[col_F]
            y_B = working_df[col_B]
            
            # Plot histograms with T-specific clipping and bins
            axes_T[i*4 + j].hist(y_F[(y_F != 0) & (y_F > T_clip_min) & (y_F < T_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
            axes_T[i*4 + j].hist(y_B[(y_B != 0) & (y_B > T_clip_min) & (y_B < T_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
            axes_T[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_T[i*4 + j].legend()
            
            if log_scale:
                axes_T[i*4 + j].set_yscale('log')  # For T values

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Grand Figure for T values, mingo0{station}\n{start_time}", fontsize=16)
    
    if save_plots:
        final_filename = f'{fig_idx}_grand_figure_T.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close(fig_T)

    # Create the grand figure for Q values
    fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_Q = axes_Q.flatten()
    
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key.replace("T", "Q")}_F_{j+1}'
            col_B = f'{key.replace("T", "Q")}_B_{j+1}'
            y_F = working_df[col_F]
            y_B = working_df[col_B]
            
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
    plt.suptitle(f"Grand Figure for Q values, mingo0{station}\n{start_time}", fontsize=16)
    
    if save_plots:
        final_filename = f'{fig_idx}_grand_figure_Q.png'
        fig_idx += 1
        
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots: plt.show()
    plt.close(fig_Q)


if create_plots or create_essential_plots:
    # Initialize figure and axes for scatter plot of Time vs Charge
    fig_TQ, axes_TQ = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_TQ = axes_TQ.flatten()

    # Iterate over each module (T1, T2, T3, T4)
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key}_F_{j+1}'  # Time F column
            col_B = f'{key}_B_{j+1}'  # Time B column
            
            y_F = working_df[col_F]  # Time values for front
            y_B = working_df[col_B]  # Time values for back
            
            charge_col_F = f'{key.replace("T", "Q")}_F_{j+1}'  # Corresponding charge column for front
            charge_col_B = f'{key.replace("T", "Q")}_B_{j+1}'  # Corresponding charge column for back
            
            charge_F = working_df[charge_col_F]  # Charge values for front
            charge_B = working_df[charge_col_B]  # Charge values for back
            
            # Apply clipping ranges to the data
            mask_F = (y_F != 0) & (y_F > T_clip_min) & (y_F < T_clip_max) & (charge_F > Q_clip_min) & (charge_F < Q_clip_max)
            mask_B = (y_B != 0) & (y_B > T_clip_min) & (y_B < T_clip_max) & (charge_B > Q_clip_min) & (charge_B < Q_clip_max)
            
            # Plot scatter plots for Time F vs Charge F and Time B vs Charge B
            axes_TQ[i*4 + j].scatter(charge_F[mask_F], y_F[mask_F], alpha=0.5, label=f'{col_F} (F)', color='green', s=1)
            axes_TQ[i*4 + j].scatter(charge_B[mask_B], y_B[mask_B], alpha=0.5, label=f'{col_B} (B)', color='orange', s=1)
            
            # Plot threshold lines for time and charge
            axes_TQ[i*4 + j].axhline(y=T_F_left_pre_cal, color='red', linestyle='--', label='T_left_pre_cal')
            axes_TQ[i*4 + j].axhline(y=T_F_right_pre_cal, color='blue', linestyle='--', label='T_right_pre_cal')
            axes_TQ[i*4 + j].axvline(x=Q_F_left_pre_cal, color='red', linestyle='--', label='Q_left_pre_cal')
            axes_TQ[i*4 + j].axvline(x=Q_F_right_pre_cal, color='blue', linestyle='--', label='Q_right_pre_cal')
            
            axes_TQ[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_TQ[i*4 + j].legend()

    # Adjust the layout and title
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f"Scatter Plot for T vs Q values, mingo0{station}\n{start_time}", fontsize=16)

    # Save the plot
    if save_plots:
        final_filename = f'{fig_idx}_scatter_plot_TQ_filtered.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    # Show the plot if requested
    if show_plots:
        plt.show()

    # Close the plot to avoid excessive memory usage
    plt.close(fig_TQ)


if self_trigger:
    if create_plots or create_essential_plots:
        # Initialize figure and axes for scatter plot of Time vs Charge
        fig_TQ, axes_TQ = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
        axes_TQ = axes_TQ.flatten()

        # Iterate over each module (T1, T2, T3, T4)
        for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
            for j in range(4):
                col_F = f'{key}_F_{j+1}'  # Time F column
                col_B = f'{key}_B_{j+1}'  # Time B column
                
                y_F = working_st_df[col_F]  # Time values for front
                y_B = working_st_df[col_B]  # Time values for back
                
                charge_col_F = f'{key.replace("T", "Q")}_F_{j+1}'  # Corresponding charge column for front
                charge_col_B = f'{key.replace("T", "Q")}_B_{j+1}'  # Corresponding charge column for back
                
                charge_F = working_st_df[charge_col_F]  # Charge values for front
                charge_B = working_st_df[charge_col_B]  # Charge values for back
                
                # Apply clipping ranges to the data
                mask_F = (y_F != 0) & (y_F > T_clip_min_ST) & (y_F < T_clip_max_ST) & (charge_F > Q_clip_min_ST) & (charge_F < Q_clip_max_ST)
                mask_B = (y_B != 0) & (y_B > T_clip_min_ST) & (y_B < T_clip_max_ST) & (charge_B > Q_clip_min_ST) & (charge_B < Q_clip_max_ST)
                
                # Plot scatter plots for Time F vs Charge F and Time B vs Charge B
                axes_TQ[i*4 + j].scatter(charge_F[mask_F], y_F[mask_F], alpha=0.5, label=f'{col_F} (F)', color='green', s=1)
                axes_TQ[i*4 + j].scatter(charge_B[mask_B], y_B[mask_B], alpha=0.5, label=f'{col_B} (B)', color='orange', s=1)
                
                # Plot threshold lines for time and charge
                axes_TQ[i*4 + j].axhline(y=T_F_left_pre_cal_ST, color='red', linestyle='--', label='T_left_pre_cal_ST')
                axes_TQ[i*4 + j].axhline(y=T_F_right_pre_cal_ST, color='blue', linestyle='--', label='T_right_pre_cal_ST')
                axes_TQ[i*4 + j].axvline(x=Q_F_left_pre_cal_ST, color='red', linestyle='--', label='Q_left_pre_cal_ST')
                axes_TQ[i*4 + j].axvline(x=Q_F_right_pre_cal_ST, color='blue', linestyle='--', label='Q_right_pre_cal_ST')
                
                axes_TQ[i*4 + j].set_title(f'{col_F} vs {col_B}')
                axes_TQ[i*4 + j].legend()

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"SELF TRIGGER. Scatter Plot for T vs Q values, mingo0{station}\n{start_time}", fontsize=16)
        if save_plots:
            final_filename = f'{fig_idx}_scatter_plot_TQ_filtered_ST.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots:
            plt.show()
        plt.close(fig_TQ)
    

# -----------------------------------------------------------------------------
# Comprobation of emptiness of the columns
# -----------------------------------------------------------------------------

# Count the number of nonzero values in each column
nonzero_counts = (working_df != 0).sum()

# Identify columns with fewer than 100 nonzero values
low_value_cols = nonzero_counts[nonzero_counts < 100].index.tolist()

if low_value_cols:
    print(f"Warning: The following columns contain fewer than 100 nonzero values and may require review: {low_value_cols}")
    print("Rejecting file due to insufficient data.")

    # Move the file to the error directory
    final_path = os.path.join(base_directories["error_directory"], file_name)
    print(f"Moving {file_path} to the error directory {final_path}...")
    shutil.move(file_path, final_path)
    now = time.time()
    os.utime(final_path, (now, now))
    sys.exit(1)


if time_window_filtering:
    
    print("----------------------------------------------------------------------")
    print("-------------------- Time window filtering (1/3) ---------------------")
    print("----------------------------------------------------------------------")
    
    for key in ['T1', 'T2', 'T3', 'T4']:
        T_F_cols = [f'{key}_F_{i+1}' for i in range(4)]
        T_B_cols = [f'{key}_B_{i+1}' for i in range(4)]

        T_F = working_df[T_F_cols].values
        T_B = working_df[T_B_cols].values

        new_cols = {}
        for i in range(4):
            new_cols[f'{key}_time_OG_sum_{i+1}'] = (T_F[:, i] + T_B[:, i]) / 2

        working_df = pd.concat([working_df, pd.DataFrame(new_cols, index=working_df.index)], axis=1)
        
    # Pre removal of outliers
    spread_results = []
    for original_tt in sorted(working_df["original_tt"].unique()):
        filtered_df = working_df[working_df["original_tt"] == original_tt].copy()
        T_sum_columns_tt = filtered_df.filter(regex='_time_OG_sum_').columns
        t_sum_spread_tt = filtered_df[T_sum_columns_tt].apply(lambda row: np.ptp(row[row != 0]) if np.any(row != 0) else np.nan, axis=1)
        filtered_df["T_sum_spread_OG"] = t_sum_spread_tt
        spread_results.append(filtered_df)
    spread_df = pd.concat(spread_results, ignore_index=True)

   
    if create_essential_plots or create_plots:
        fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=False)
        axs = axs.flatten()
        for i, tt in enumerate(sorted(spread_df["original_tt"].unique())):
            subset = spread_df[spread_df["original_tt"] == tt]
            v = subset["T_sum_spread_OG"].dropna()
            v = v[v < coincidence_window_og_ns * 3]
            axs[i].hist(v, bins=100, alpha=0.7)
            axs[i].set_title(f"TT = {tt}")
            axs[i].set_xlabel("ΔT (ns)")
            axs[i].set_ylabel("Events")
            axs[i].axvline(x=coincidence_window_og_ns, color='red', linestyle='--', label='Time coincidence window')
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
    def zero_outlier_tsum(row, threshold=coincidence_window_og_ns):
        t_sum_cols = [col for col in row.index if 'T' in col]
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
        T_sum_columns_tt = filtered_df.filter(regex='_time_OG_sum_').columns
        t_sum_spread_tt = filtered_df[T_sum_columns_tt].apply(lambda row: np.ptp(row[row != 0]) if np.any(row != 0) else np.nan, axis=1)
        filtered_df["T_sum_spread_OG"] = t_sum_spread_tt
        spread_results.append(filtered_df)
    spread_df = pd.concat(spread_results, ignore_index=True)

   
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
            axs[i].axvline(x=coincidence_window_og_ns, color='red', linestyle='--', label='Time coincidence window')# Logscale
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


if os.path.exists(temp_file):
    print("Removing temporary file...")
    os.remove(temp_file)



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
                



# Path to save the cleaned dataframe
# Create output directory if it does not exist /home/mingo/DATAFLOW_v3/MASTER/STAGE_1/EVENT_DATA/STEP_1/TASK_1/DONE/
os.makedirs(f"{output_directory}", exist_ok=True)
OUT_PATH = f"{output_directory}/cleaned_{basename_no_ext}.h5"
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






# If Q*_F_* and Q*_B_* are zero for all cases, remove the row
Q_F_cols = _collect_columns(working_df.columns, Q_FRONT_PATTERN)
Q_B_cols = _collect_columns(working_df.columns, Q_BACK_PATTERN)
working_df = working_df[(working_df[Q_F_cols] != 0).any(axis=1) & (working_df[Q_B_cols] != 0).any(axis=1)]


print(f"Original number of events in the dataframe: {original_number_of_events}")
# Final number of events
final_number_of_events = len(working_df)
print(f"Final number of events in the dataframe: {final_number_of_events}")




# ----------------------------------------------------------------------------------
# Count the number of non-zero entries per channel in the whole dataframe ----------
# ----------------------------------------------------------------------------------

# Count per each column the number of non-zero entries and save it in a column of
# global_variables called TX_F_Y_entries or TX_B_Y_entries

# Count for main dataframe (non-self-trigger)
for key, idx_range in column_indices.items():
    for i in range(1, len(idx_range) + 1):
        colname = f"{key}_{i}"
        count = (working_df[colname] != 0).sum()
        global_var_name = f"{key}_{i}_entries_final"
        global_variables[global_var_name] = count





# Data purity
data_purity = final_number_of_events / original_number_of_events * 100
print(f"Data purity is {data_purity:.2f}%")


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
working_df.to_parquet(OUT_PATH, engine="pyarrow", compression="zstd", index=False)
print(f"Cleaned dataframe saved to: {OUT_PATH}")


# Move the original datafile to COMPLETED -------------------------------------
print("Moving file to COMPLETED directory...")

if user_file_selection == False:
    shutil.move(file_path, completed_file_path)
    now = time.time()
    os.utime(completed_file_path, (now, now))
    print("************************************************************")
    print(f"File moved from\n{file_path}\nto:\n{completed_file_path}")
    print("************************************************************")
