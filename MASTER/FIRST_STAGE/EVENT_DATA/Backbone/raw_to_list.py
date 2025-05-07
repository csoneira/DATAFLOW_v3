#!/usr/bin/env python3
# -*- coding: utf-8 -*-

stratos_save = True

fast_mode = False # Do not iterate TimTrack, neither save figures, etc.
debug_mode = False # Only 10000 rows with all detail
last_file_test = False

alternative_fitting = True

"""
Created on Thu Jun 20 09:15:33 2024

@author: csoneira@ucm.es
"""

# globals().clear()

import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
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
station = sys.argv[1]
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
    "stratos_list_events_directory": os.path.join(home_directory, "STRATOS_XY_DIRECTORY"),
    
    "base_plots_directory": os.path.join(raw_to_list_working_directory, "PLOTS"),
    
    "pdf_directory": os.path.join(raw_to_list_working_directory, "PLOTS/PDF_DIRECTORY"),
    "base_figure_directory": os.path.join(raw_to_list_working_directory, "PLOTS/FIGURE_DIRECTORY"),
    "figure_directory": os.path.join(raw_to_list_working_directory, f"PLOTS/FIGURE_DIRECTORY/FIGURES_EXEC_ON_{date_execution}"),
    
    "list_events_directory": os.path.join(base_directory, "LIST_EVENTS_DIRECTORY"),
    "full_list_events_directory": os.path.join(base_directory, "FULL_LIST_EVENTS_DIRECTORY"),
    
    "ancillary_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY"),
    
    "empty_files_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY/EMPTY_FILES"),
    "rejected_files_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY/REJECTED_FILES"),
    "temp_files_directory": os.path.join(raw_to_list_working_directory, "ANCILLARY/TEMP_FILES"),
    
    "unprocessed_directory": os.path.join(raw_to_list_working_directory, "RAW_TO_LIST_FILES/UNPROCESSED_DIRECTORY"),
    "error_directory": os.path.join(raw_to_list_working_directory, "RAW_TO_LIST_FILES/ERROR_DIRECTORY"),
    "processing_directory": os.path.join(raw_to_list_working_directory, "RAW_TO_LIST_FILES/PROCESSING_DIRECTORY"),
    "completed_directory": os.path.join(raw_to_list_working_directory, "RAW_TO_LIST_FILES/COMPLETED_DIRECTORY"),
    
    "raw_directory": os.path.join(raw_working_directory, "."),
}

# Create ALL directories if they don't already exist
for directory in base_directories.values():
    os.makedirs(directory, exist_ok=True)

csv_path = os.path.join(base_directory, "calibrations.csv")

# Move files from RAW to RAW_TO_LIST/RAW_TO_LIST_FILES/UNPROCESSED,
# ensuring that only files not already in UNPROCESSED, PROCESSING,
# or COMPLETED are moved:

raw_directory = base_directories["raw_directory"]
unprocessed_directory = base_directories["unprocessed_directory"]
error_directory = base_directories["error_directory"]
stratos_list_events_directory = base_directories["stratos_list_events_directory"]
processing_directory = base_directories["processing_directory"]
completed_directory = base_directories["completed_directory"]

empty_files_directory = base_directories["empty_files_directory"]
rejected_files_directory = base_directories["rejected_files_directory"]
temp_files_directory = base_directories["temp_files_directory"]

raw_files = set(os.listdir(raw_directory))
unprocessed_files = set(os.listdir(unprocessed_directory))
processing_files = set(os.listdir(processing_directory))
completed_files = set(os.listdir(completed_directory))

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

# Files to move: in RAW but not in UNPROCESSED, PROCESSING, or COMPLETED
files_to_move = raw_files - unprocessed_files - processing_files - completed_files

# Copy files to UNPROCESSED
for file_name in files_to_move:
    src_path = os.path.join(raw_directory, file_name)
    dest_path = os.path.join(unprocessed_directory, file_name)
    try:
        shutil.move(src_path, dest_path)
        print(f"Move {file_name} to UNPROCESSED directory.")
    except Exception as e:
        print(f"Failed to copy {file_name}: {e}")


# Erase all files in the figure_directory
figure_directory = base_directories["figure_directory"]
files = os.listdir(figure_directory)

if files:  # Check if the directory contains any files
    print("Removing all files in the figure_directory...")
    for file in files:
        os.remove(os.path.join(figure_directory, file))


# Define input file path -----------------------------------------------------
input_file_config_path = os.path.join(station_directory, f"input_file_mingo0{station}.csv")

if os.path.exists(input_file_config_path):
    print("Searching input configuration file:", input_file_config_path)
    
    # It is a csv
    input_file = pd.read_csv(input_file_config_path, skiprows=1)
    
    print("Input configuration file found.")
    exists_input_file = True
    
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


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Header ----------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Execution options -----------------------------------------------------------
# -----------------------------------------------------------------------------

# Plots and savings -------------------------
crontab_execution = True
create_plots = False
create_essential_plots = True
save_plots = True
show_plots = False
create_pdf = True
limit = False
limit_number = 10000
number_of_time_cal_figures = 3
save_calibrations = True
save_full_data = True
presentation = False
presentation_plots = False
force_replacement = True # Creates a new datafile even if there is already one that looks complete
article_format = False

# Charge calibration to fC -------------------------
calibrate_charge = True

# Charge front-back --------------------------------
charge_front_back = True

# Y position ---------------------------------------
y_position_complex_method = False
uniform_y_method = True

# Slewing correction -------------------------------
slewing_correction = True

# Time calibration ---------------------------------
time_calibration = True

# RPC variables ------------------------------------
weighted = False

# TimTrack -----------------------------------------
fixed_speed = False
res_ana_removing_planes = False
timtrack_iteration = False
number_of_TT_executions = 2
plot_three_planes = False
residual_plots = True

if fast_mode:
    print('Working in fast mode.')
    residual_plots = False
    timtrack_iteration = False
    time_calibration = False
    charge_front_back = False
    create_plots = False
    # save_full_data = False
    limit = False
    limit_number = 10000
    
if debug_mode:
    print('Working in debug mode.')
    residual_plots = True
    timtrack_iteration = False
    time_calibration = False
    charge_front_back = False
    create_plots = True
    # save_full_data = False
    limit = True
    limit_number = 10000

# -----------------------------------------------------------------------------
# Filters ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

# General -------------------------

# Pre-cal Front & Back ---------
if debug_mode:
    T_F_left_pre_cal = -500
    T_F_right_pre_cal = 500

    T_B_left_pre_cal = -500
    T_B_right_pre_cal = 500

    Q_F_left_pre_cal = -500
    Q_F_right_pre_cal = 500

    Q_B_left_pre_cal = -500
    Q_B_right_pre_cal = 500
else:
    T_F_left_pre_cal = -200
    T_F_right_pre_cal = 100

    T_B_left_pre_cal = -200
    T_B_right_pre_cal = 100

    Q_F_left_pre_cal = 50
    Q_F_right_pre_cal = 500

    Q_B_left_pre_cal = 50
    Q_B_right_pre_cal = 500

T_left_side = T_F_left_pre_cal
T_right_side = T_F_right_pre_cal

Q_left_side = Q_F_left_pre_cal
Q_right_side = 150

# Pre-cal Sum & Diff ---------
# Qsum
Q_left_pre_cal = -500
Q_right_pre_cal = 500
# Qdif
Q_diff_pre_cal_threshold = 500
# Tsum
T_sum_left_pre_cal = -500 # it was -130 for mingo01 etc but for mingo03 it had to be changed
T_sum_right_pre_cal = 500
# Tdif
T_diff_pre_cal_threshold = 20

# Post-calibration ---------
# Qsum
Q_sum_left_cal = -30
Q_sum_right_cal = 1000
# Qdif
Q_diff_cal_threshold = 10
Q_diff_cal_threshold_FB = 5
# Tsum
# ...
# Tdif
T_diff_cal_threshold = 1


# Once calculated the RPC variables
# Tsum
T_sum_RPC_left = -300
T_sum_RPC_right = 300
# Tdiff
T_diff_RPC_left = -0.8
T_diff_RPC_right = 0.8
# Qsum
Q_RPC_left = 0
Q_RPC_right = 1000
# Y pos
Y_RPC_left = -200 # -150
Y_RPC_right = 200 # 150

# TimTrack filter -------------------------
pos_filter = 700
proj_filter = 2
t0_left_filter = T_sum_RPC_left
t0_right_filter = T_sum_RPC_right
slowness_filter_left = -0.01 # -0.01
slowness_filter_right = 0.025 # 0.025
charge_strip_left_filter = -1e6
charge_strip_right_filter = 1e6
charge_event_left_filter = -1e6
charge_event_right_filter = 1e6

res_ystr_filter = 120
res_tsum_filter = 2
res_tdif_filter = 0.4

ext_res_ystr_filter = 120
ext_res_tsum_filter = 2
ext_res_tdif_filter = 1

# -----------------------------------------------------------------------------
# Calibrations ----------------------------------------------------------------
# -----------------------------------------------------------------------------

# General
calibrate_strip_T_percentile = 5
calibrate_strip_Q_percentile = 5
calibrate_strip_Q_FB_percentile = 5

# Time sum
CRT_gaussian_fit_quantile = 0.03
strip_time_diff_bound = 10
time_coincidence_window = 7

# Front-back charge
distance_sum_charges_left_fit = -5
distance_sum_charges_right_fit = 200
distance_diff_charges_up_fit = 10
distance_diff_charges_low_fit = -10
distance_sum_charges_plot = 800
front_back_fit_threshold = 4 # It was 1.4

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
    [0.0, -0.3886208, -0.53020947, 0.33711737],
    [-0.80494094, -0.68836069, -2.01289387, -1.13481931],
    [-0.23899338, -0.51373738, 0.50845317, 0.11685095],
    [0.33586285, 1.08329847, 0.91410244, 0.58815813]
])

# -----------------------------------------------------------------------------
# Variables to modify ---------------------------------------------------------
# -----------------------------------------------------------------------------

beta = 1 # Given the last fitting of slowness

# Y position parameters
transf_exp = 1
induction_section = 0  # 40 In mm. Width of the induction section for all strips
y_pos_threshold = 10000  # In mm. Not a real distance. Adjust this value as needed for "closeness"

# -----------------------------------------------------------------------------
# Variables to not touch unless necessary -------------------------------------
# -----------------------------------------------------------------------------

Q_sum_color = 'orange'
Q_diff_color = 'red'
T_sum_color = 'blue'
T_diff_color = 'green'

fig_idx = 1
plot_list = []

# Front-back charge
output_order = 0
degree_of_polynomial = 4

# X ----------------------------
strip_length = 300

# Y ----------------------------
def y_pos(y_width):
    return np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2

y_widths = [np.array([63, 63, 63, 98]), np.array([98, 63, 63, 63])]  # P1-P3 and P2-P4 widths
y_pos_T = [y_pos(y_widths[0]), y_pos(y_widths[1])]
y_width_P1_and_P3 = y_widths[0]
y_width_P2_and_P4 = y_widths[1]
y_pos_P1_and_P3 = y_pos(y_width_P1_and_P3)
y_pos_P2_and_P4 = y_pos(y_width_P2_and_P4)

# Miscelanous ----------------------------
c_mm_ns = c/1000000
muon_speed = beta * c_mm_ns
strip_speed = 2/3 * c_mm_ns # 200 mm/ns
tdiff_to_x = strip_speed # Factor to transform t_diff to X

# Timtrack parameters --------------------
vc    = beta * c_mm_ns #mm/ns
sc    = 1/vc
ss    = 1/strip_speed # slowness of the signal in the strip
cocut = 1  # convergence cut
d0    = 10 # initial value of the convergence parameter 
nplan = 4
lenx  = strip_length

anc_sy = 25 # 2.5 cm
anc_sts = 0.4 # 400ps
anc_std = 0.1 # 2 cm
anc_sx = strip_speed * anc_std # 2 cm
anc_sz = 10 # 5 cm


# -----------------------------------------------------------------------------
# Some variables that define the analysis, define a dictionary with the variables:
# 'discarded_by_time_window', 'one_side_events', 'purity_of_data'
# -----------------------------------------------------------------------------
global_variables = {
    'CRP_avg': 0,
    'discarded_by_time_window_percentage': 0,
    'one_side_events': 0,
    'purity_of_data_percentage': 0,
    'unc_y': anc_sy,
    'unc_tsum': anc_sts,
    'unc_tdif': anc_std
}

# Modify discarded_by_time_window entry
global_variables['discarded_by_time_window'] = 1

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Function definition ---------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Calibration functions
# def calibrate_strip_T(column):
#     q = calibrate_strip_T_percentile
#     mask = (abs(column) < T_diff_pre_cal_threshold)
#     column = column[mask]
#     column = column[column != 0]
#     column = column[(np.percentile(column, q) < column) & (column < np.percentile(column, 100 - q))]
#     column = column[(np.percentile(column, q) < column) & (column < np.percentile(column, 100 - q))]
#     column = column[(np.percentile(column, q) < column) & (column < np.percentile(column, 100 - q))]
#     offset = np.median([np.min(column), np.max(column)])
#     return offset


def calibrate_strip_T(column, num_bins=100):
    """
    Calibrates a given column of T values by filtering and determining an offset.

    Parameters:
        column (numpy.ndarray): Input array of T values.
        num_bins (int): Number of bins to use in the histogram.

    Returns:
        float: Calculated offset.
    """
    
    T_rel_th = 0.9
    
    # Apply mask to filter values within the threshold
    mask = (np.abs(column) < T_diff_pre_cal_threshold)
    column = column[mask]
    
    # Remove zero values
    column = column[column != 0]
    
    # Calculate histogram
    counts, bin_edges = np.histogram(column, bins=num_bins)
    
    # Find the maximum number of counts in any bin
    max_counts = np.max(counts)
    
    # Identify bins with counts above the relative threshold
    valid_bins = (counts > T_rel_th * max_counts)
    
    # Filter the original column values based on the valid bins
    column_filt = []
    for i, valid in enumerate(valid_bins):
        if valid:
            # Include values within the range of this bin
            bin_min = bin_edges[i]
            bin_max = bin_edges[i + 1]
            column_filt.extend(column[(column >= bin_min) & (column < bin_max)])
    column_filt = np.array(column_filt)
    
    # Calculate the offset using the mean of the filtered values
    offset = np.mean([np.min(column_filt), np.max(column_filt)])
    
    return offset


def calibrate_strip_T_diff(T_F, T_B):
    """
    Calibrates a given column of T values by filtering and determining an offset.

    Parameters:
        column (numpy.ndarray): Input array of T values.
        num_bins (int): Number of bins to use in the histogram.

    Returns:
        float: Calculated offset.
    """
    
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
    
    T_diff = ( T_F - T_B ) / 2
    
    # print("Zeroes:")
    # print(len(T_diff[T_diff == 0]))
    
    # ------------------------------------------------------------------------------
    
    T_rel_th = 0.1
    abs_th = 1
    
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
    
    # print(plateau_left)
    # print(plateau_right)
    
    # Calculate the offset using the mean of the filtered values
    offset = ( plateau_left + plateau_right ) / 2
    
    return offset


def calibrate_strip_Q_pedestal(Q_ch, T_ch, Q_other):
    """
    Calibrate the pedestal offset for the charge distribution (Q_ch) by finding
    the first bin of the longest subset of bins with at least one count.

    Parameters:
        Q_ch (numpy.ndarray): Array of charge values for the channel.
        num_bins (int): Number of bins to use for the histogram.

    Returns:
        float: Offset value to bring the distribution to zero.
    """
    
    # First let's tale good values of Time, we want to avoid outliers that might confuse the charge pedestal calibration
    cond = (T_ch != 0) & (T_ch > T_left_side) & (T_ch < T_right_side)
    T_ch = T_ch[cond]
    Q_ch = Q_ch[cond]
    Q_other = Q_other[cond]
    
    # Condition based on the charge difference: it cannot be too high
    Q_dif = Q_ch - Q_other
    percentile = 5
    cond = ( Q_dif > np.percentile(Q_dif, percentile) ) & ( Q_dif < np.percentile(Q_dif, 100 - percentile ) )
    T_ch = T_ch[cond]
    Q_ch = Q_ch[cond]
    
    counts, bin_edges = np.histogram(T_ch, bins='auto')
    max_counts = np.max(counts)
    min_counts = np.min(counts[counts > 0])
    threshold = max_counts / 10**1.5
    indices_above_threshold = np.where(counts > threshold)[0]

    if indices_above_threshold.size > 0:
        min_bin_edge = bin_edges[indices_above_threshold[0]]
        max_bin_edge = bin_edges[indices_above_threshold[-1] + 1]  # +1 to get the upper edge of the last bin
        # print(f"Minimum bin edge: {min_bin_edge}")
        # print(f"Maximum bin edge: {max_bin_edge}")
    else:
        print("No bins have counts above the threshold; Q pedestal calibration.")
        threshold = (min_counts + max_counts) / 2
        indices_above_threshold = np.where(counts > threshold)[0]
        min_bin_edge = bin_edges[indices_above_threshold[0]]
        max_bin_edge = bin_edges[indices_above_threshold[-1] + 1]
    
    Q_ch = Q_ch[(T_ch > min_bin_edge) & (T_ch < max_bin_edge)]
    
    # 5% of the maximum count
    rel_th = 0.015
    rel_th_cal = 0.3
    abs_th = 3
    q_quantile = 0.4 # percentile
    
    # First take the values that are not zero
    Q_ch = Q_ch[Q_ch != 0]
    
    # Remove the values that are not in (50,500)
    Q_ch = Q_ch[(Q_ch > Q_left_side) & (Q_ch < Q_right_side)]
    
    # Quantile filtering
    Q_ch = Q_ch[Q_ch > np.percentile(Q_ch, q_quantile)]
    
    # num_bins = int(len(Q_ch) / 100)
    
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
    Q_ch_cal = Q_ch_cal[(Q_ch_cal > -1) & (Q_ch_cal < 2)]
    
    # Q_ch_cal = Q_ch_cal[Q_ch_cal > np.percentile(Q_ch_cal, q_quantile)]
    
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
    
    # print(offset_cal)
    
    pedestal = offset + offset_cal
    
    pedestal = offset
    return pedestal


# def calibrate_strip_Q(Q_sum):
#     q = calibrate_strip_Q_percentile
#     mask_Q = (Q_sum != 0)
#     Q_sum = Q_sum[mask_Q]
#     mask_Q = (Q_sum > Q_left_pre_cal) & (Q_sum < Q_right_pre_cal)
#     Q_sum = Q_sum[mask_Q]
#     Q_sum = Q_sum[Q_sum > np.percentile(Q_sum, q)]
#     mean = np.mean(Q_sum)
#     std = np.std(Q_sum)
#     Q_sum = Q_sum[ abs(Q_sum - mean) < std ]
#     offset = np.min(Q_sum)
#     return offset

def calibrate_strip_Q_FB(Q_F, Q_B):
    q = calibrate_strip_Q_FB_percentile
    
    mask_Q = (Q_F != 0)
    Q_F = Q_F[mask_Q]
    mask_Q = (Q_F > Q_left_pre_cal) & (Q_F < Q_right_pre_cal)
    Q_F = Q_F[mask_Q]
    Q_F = Q_F[Q_F > np.percentile(Q_F, q)]
    mean = np.mean(Q_F)
    std = np.std(Q_F)
    Q_F = Q_F[ abs(Q_F - mean) < std ]
    offset_F = np.min(Q_F)
    
    mask_Q = (Q_B != 0)
    Q_B = Q_B[mask_Q]
    mask_Q = (Q_B > Q_left_pre_cal) & (Q_B < Q_right_pre_cal)
    Q_B = Q_B[mask_Q]
    Q_B = Q_B[Q_B > np.percentile(Q_B, q)]
    mean = np.mean(Q_B)
    std = np.std(Q_B)
    Q_B = Q_B[ abs(Q_B - mean) < std ]
    offset_B = np.min(Q_B)
    
    return (offset_F - offset_B) / 2

import builtins
enumerate = builtins.enumerate

def polynomial(x, *coeffs):
    return sum(c * x**i for i, c in enumerate(coeffs))

def scatter_2d_and_fit(xdat, ydat, title, x_label, y_label, name_of_file):
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
        
        # (16,6) was very nice
        if article_format:
            ww = (10.84, 4)
        else:
            ww = (13.33, 5)
            
        plt.figure(figsize=ww)  # Use plt.subplots() to create figure and axis    
        plt.scatter(xdat_plot, ydat_plot, s=1, label="Original data points")
        # plt.scatter(xdat_pre_fit, ydat_pre_fit, s=1, color="magenta", label="Points for prefitting")
        plt.scatter(xdat_fit, ydat_fit, s=1, color="orange", label="Points for fitting")
        plt.scatter(x_final, y_final, s=1, color="green", label="Calibrated points")
        plt.plot(x_fit, y_fit, 'r-', label='Polynomial Fit: ' + ' '.join([f'a{i}={coeff:.2g}' for i, coeff in enumerate(coeffs[::-1])]))
        
        if not article_format:
            plt.title(f"Fig. {output_order}, {title}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim([-5, 400])
        plt.ylim([-11, 11])
        
        plt.grid()
        plt.legend(markerscale=5)  # Increase marker scale by 5 times
        
        plt.tight_layout()
        # plt.savefig(f"{output_order}_{name_of_file}.png", format="png")
        
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
        
        # (16,6) was very nice
        if article_format:
            ww = (10.84, 4)
        else:
            ww = (13.33, 5)
            
        plt.figure(figsize=ww)  # Use plt.subplots() to create figure and axis    
        plt.scatter(xdat_plot, ydat_plot, s=1, label="Original data points")
        # plt.scatter(xdat_pre_fit, ydat_pre_fit, s=1, color="magenta", label="Points for prefitting")
        plt.scatter(xdat_fit, ydat_fit, s=1, color="orange", label="Points for fitting")
        plt.scatter(x_final, y_final, s=1, color="green", label="Calibrated points")
        plt.plot(x_fit, y_fit, 'r-', label='Polynomial Fit: ' + ' '.join([f'a{i}={coeff:.2g}' for i, coeff in enumerate(coeffs[::-1])]))
        
        if not article_format:
            plt.title(f"Fig. {output_order}, {title}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim([-5, 200])
        plt.ylim([-11, 11])
        
        plt.grid()
        plt.legend(markerscale=5)  # Increase marker scale by 5 times
        
        plt.tight_layout()
        # plt.savefig(f"{output_order}_{name_of_file}.png", format="png")
        
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

def summary_skew(vdat):
    # Calculate the 5th and 95th percentiles
    try:
        percentile_left = np.percentile(vdat, 20)
        percentile_right = np.percentile(vdat, 80)
    except IndexError:
        print("Problem with indices")
        # print(vector)
        
    # Filter values inside the 5th and 95th percentiles
    vdat = [x for x in vdat if percentile_left <= x <= percentile_right]
    mean = np.mean(vdat)
    std = np.std(vdat)
    skewness = skew(vdat)
    return f"mean = {mean:.2g}, std = {std:.2g}, skewness = {skewness:.2g}"

def summary(vector):
    quantile_left = CRT_gaussian_fit_quantile * 100
    quantile_right = 100 - CRT_gaussian_fit_quantile * 100
    
    vector = np.array(vector)  # Convert list to NumPy array
    strip_time_diff_bound = 10
    cond = (vector > -strip_time_diff_bound) & (vector < strip_time_diff_bound)  # This should result in a boolean array
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

from scipy.stats import norm

def hist_1d(vdat, bin_number, title, axis_label, name_of_file):
    global fig_idx

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    
    # Create histogram without plotting it
    # counts, bins, _ = ax.hist(vdat, bins=bin_number, alpha=0.5, color="red",
    #                           label=f"All hits, {len(vdat)} events, {summary_skew(vdat)}", density=False)
    
    vdat = np.array(vdat)  # Convert list to NumPy array
    strip_time_diff_bound = 10
    cond = (vdat > -strip_time_diff_bound) & (vdat < strip_time_diff_bound)  # This should result in a boolean array
    vdat = vdat[cond]
    
    counts, bins, _ = ax.hist(vdat, bins=bin_number, alpha=0.5, color="red",
                              label=f"All hits, {len(vdat)} events", density=False)
    
    # Calculate bin centers for fitting the Gaussian
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Fit a Gaussian
    h1_q = CRT_gaussian_fit_quantile
    lower_bound = np.quantile(vdat, h1_q)
    upper_bound = np.quantile(vdat, 1 - h1_q)
    
    cond = (vdat > lower_bound) & (vdat < upper_bound)  # This should result in a boolean array
    vdat = vdat[cond]
    
    mu, std = norm.fit(vdat)

    # Plot the Gaussian fit
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
    
    
# Define the combined function to plot histograms and optionally fit Gaussian
# def plot_histograms_and_gaussian(df, columns, title, figure_number, quantile=0.99, fit_gaussian=False):
#     global fig_idx
#     nrows, ncols = (2, 3) if figure_number == 1 else (3, 4)
    
#     fig, axs = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows), constrained_layout=True)
#     axs = axs.flatten()

#     # Define Gaussian function
#     def gaussian(x, mu, sigma, amplitude):
#         return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

#     # Plot histograms and fit Gaussian if needed
#     for i, col in enumerate(columns):
#         data = df[col].values
#         data = data[data != 0]  # Filter out zero values

#         # Check if data is empty or has fewer points than needed
#         if len(data) == 0:
#             # Leave plot empty if no data
#             axs[i].text(0.5, 0.5, "No data", transform=axs[i].transAxes, ha='center', va='center', color='gray')
#             continue

#         # Plot histogram
#         hist_data, bin_edges, _ = axs[i].hist(data, bins='auto', alpha=0.75, label='Data')
#         axs[i].set_title(col)
#         axs[i].set_xlabel('Value')
#         axs[i].set_ylabel('Frequency')

#         # Fit Gaussian if needed and if there's enough data
#         if fit_gaussian and len(data) >= 10:
#             try:
#                 # Quantile filtering
#                 lower_bound, upper_bound = np.quantile(data, [(1 - quantile), quantile])
#                 filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

#                 if len(filtered_data) < 2:  # Ensure there are enough points to fit a Gaussian
#                     axs[i].text(0.5, 0.5, "Not enough data to fit", transform=axs[i].transAxes, ha='center', va='center', color='gray')
#                     continue

#                 # Fit Gaussian to the filtered data
#                 bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#                 popt, _ = curve_fit(gaussian, bin_centers, hist_data, p0=[np.mean(filtered_data), np.std(filtered_data), max(hist_data)])
#                 mu, sigma, amplitude = popt
#                 x = np.linspace(min(filtered_data), max(filtered_data), 1000)
#                 axs[i].plot(x, gaussian(x, mu, sigma, amplitude), 'r-', label=f'Gaussian Fit\nμ={mu:.2g}, σ={sigma:.2g}')
#                 axs[i].legend()
#             except (RuntimeError, ValueError):
#                 axs[i].text(0.5, 0.5, "Fit failed", transform=axs[i].transAxes, ha='center', va='center', color='red')

#     # Remove unused subplots
#     for j in range(i + 1, len(axs)):
#         fig.delaxes(axs[j])

#     plt.suptitle(title, fontsize=16)
    
#     if save_plots:
#         final_filename = f'{fig_idx}_{title.replace(" ", "_")}.png'
#         fig_idx += 1

#         save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
#         plot_list.append(save_fig_path)
#         plt.savefig(save_fig_path, format='png')
    
#     if show_plots: plt.show()
#     plt.close()


def plot_histograms_and_gaussian(df, columns, title, figure_number, quantile=0.99, fit_gaussian=False):
    global fig_idx
    nrows, ncols = (2, 3) if figure_number == 1 else (3, 4)
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows), constrained_layout=True)
    axs = axs.flatten()

    # Define Gaussian function
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

        # Plot histogram
        hist_data, bin_edges, _ = axs[i].hist(data, bins='auto', alpha=0.75, label='Data')
        axs[i].set_title(col)
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')

        # Fit Gaussian if enabled and data is sufficient
        if fit_gaussian and len(data) >= 10:
            try:
                # Use precomputed quantile bounds
                if col in quantile_bounds:
                    lower_bound, upper_bound = quantile_bounds[col]
                    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

                if len(filtered_data) < 2:
                    axs[i].text(0.5, 0.5, "Not enough data to fit", transform=axs[i].transAxes, ha='center', va='center', color='gray')
                    continue

                # Fit Gaussian to the histogram data
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                popt, _ = curve_fit(gaussian, bin_centers, hist_data, p0=[np.mean(filtered_data), np.std(filtered_data), max(hist_data)])
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
    return True

def get_file_path(directory, file_name):
    return os.path.join(directory, file_name)


# Create ALL directories if they don't already exist
for directory in base_directories.values():
    os.makedirs(directory, exist_ok=True)

unprocessed_files = os.listdir(base_directories["unprocessed_directory"])
processing_files = os.listdir(base_directories["processing_directory"])
completed_files = os.listdir(base_directories["completed_directory"])

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

# Check the station number in the datafile
try:
    file_station_number = int(file_name[3])  # 4th character (index 3)
    if file_station_number != int(station):
        print(f'File station number is: {file_station_number}, it does not match.')
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
EXPECTED_COLUMNS = 71  # Expected number of columns

# Function to process each line
def process_line(line):
    line = re.sub(r'0000\.0000', '0', line)  # Replace '0000.0000' with '0'
    line = re.sub(r'\b0+([0-9]+)', r'\1', line)  # Remove leading zeros
    line = re.sub(r' +', ',', line.strip())  # Replace multiple spaces with a comma
    line = re.sub(r'X(202\d)', r'X\n\1', line)  # Replace X2024, X2025 with X\n202Y
    line = re.sub(r'(\w)-(\d)', r'\1 -\2', line)  # Ensure X-Y is properly spaced
    return line

# Function to check for malformed numbers (e.g., '-120.144.0')
def contains_malformed_numbers(line):
    return bool(re.search(r'-?\d+\.\d+\.\d+', line))  # Detects multiple decimal points

# Function to validate year, month, and day
def is_valid_date(values):
    try:
        year, month, day = int(values[0]), int(values[1]), int(values[2])
        if year not in {2023, 2024, 2025, 2026, 2027}:  # Check valid years
            return False
        if not (1 <= month <= 12):  # Check valid month
            return False
        if not (1 <= day <= 31):  # Check valid day
            return False
        return True
    except ValueError:  # In case of non-numeric values
        return False

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

data = pd.read_csv(temp_file, header=None, low_memory=False, nrows=limit_number if limit else None)
data = data.apply(pd.to_numeric, errors='coerce')

# Print the number of rows in input
print("*******************************************************")
print(f"Original file has {read_lines} lines.")
print(f"Processed file has {written_lines} lines.")
print("*******************************************************")

# ------------------------------------------------------------------------------------------------------

# data = pd.read_csv(file_path, sep=r'\s+', header=None, nrows=limit_number if limit else None, on_bad_lines='skip', low_memory=False)

# Assign name to the columns
data.columns = ['year', 'month', 'day', 'hour', 'minute', 'second'] + [f'column_{i}' for i in range(6, 71)]
data['datetime'] = pd.to_datetime(data[['year', 'month', 'day', 'hour', 'minute', 'second']])

# ------------------------------------------------------------------------------------------------------
# Filter 1: by date ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
print("----------------------- Filter 1: by date -----------------------")
filtered_data = data[(data['datetime'] >= left_limit_time) & (data['datetime'] <= right_limit_time)]
og_data = filtered_data.copy()
og_data = og_data.set_index('datetime')  # Set 'datetime' as index
if not isinstance(og_data.index, pd.DatetimeIndex):
    raise ValueError("The index is not a DatetimeIndex. Check 'datetime' column formatting.")

raw_data_len = len(filtered_data)
# if debug_mode:
#     print(raw_data_len)

# Print the count frequency of the values in column_6
print(filtered_data['column_6'].value_counts())
# Take only the rows in which column_6 is equal to 1
filtered_data = filtered_data[filtered_data['column_6'] == 1]

raw_data_len = len(filtered_data)
# if debug_mode:
#     print(raw_data_len)

if raw_data_len == 0:  # Use '==' for comparison
    print(filtered_data['column_6'].head())
    print("No coincidence events.")
    sys.exit()

# Note that the middle between start and end time could also be taken. This is for calibration storage.
datetime_value = filtered_data['datetime'][0]
# Take the last datetime value
end_datetime_value = filtered_data['datetime'].iloc[-1]
start_time = datetime_value
end_time = end_datetime_value
datetime_str = str(datetime_value)
save_filename_suffix = datetime_str.replace(' ', "_").replace(':', ".").replace('-', ".")

# -------------------------------------------------------------------------------
# Input file and data managing to select configuration
# -------------------------------------------------------------------------------


print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
if exists_input_file:
    # Ensure `start` and `end` columns are in datetime format
    input_file["start"] = pd.to_datetime(input_file["start"])
    input_file["end"] = pd.to_datetime(input_file["end"])
    
    input_file["end"].fillna(pd.to_datetime('now'), inplace=True)
    
    # Filter matching configurations
    matching_confs = input_file[ (input_file["start"] <= start_time) & (input_file["end"] >= end_time) ]

    # Select the first matching configuration if available
    if not matching_confs.empty:
        if len(matching_confs) > 1:
            print(f"Warning:\nMultiple configurations match the date range\n{start_time} to {end_time}.\nTaking the first one.")
        
        selected_conf = matching_confs.iloc[0]
        print(f"Selected configuration: {selected_conf['conf']}")

        # Extract z_1 to z_4 values
        z_positions = np.array([selected_conf.get(f"P{i}", np.nan) for i in range(1, 5)])

    else:
        print("Error: No matching configuration found for the given date range. Using default z_positions.")
        z_positions = np.array([0, 150, 300, 450])  # In mm
        
else:
    print("Error: No input file. Using default z_positions.")
    z_positions = np.array([0, 150, 300, 450])  # In mm

# Print the resulting z_positions
z_positions = z_positions - z_positions[0]
print(f"Z positions: {z_positions}")

print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")

# -------------------------------------------------------------------------------

print("----------------------------------------------------------")
print("----------------------------------------------------------")
print(f"----------- Starting date is {save_filename_suffix} --------------")
print("----------------------------------------------------------")
print("----------------------------------------------------------")

# Defining the directories that will store the data
save_full_filename = f"full_list_events_{save_filename_suffix}.txt"
save_filename = f"list_events_{save_filename_suffix}.txt"
save_pdf_filename = f"pdf_{save_filename_suffix}.pdf"

save_list_path = os.path.join(base_directories["list_events_directory"], save_filename)
save_full_path = os.path.join(base_directories["full_list_events_directory"], save_full_filename)
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
columns_data = {'datetime': filtered_data['datetime'].values}
for key, idx_range in column_indices.items():
    for i, col_idx in enumerate(idx_range):
        column_name = f'{key}_{i+1}'
        columns_data[column_name] = filtered_data.iloc[:, col_idx].values

# Create a DataFrame from the columns data
final_df = pd.DataFrame(columns_data)

if debug_mode:
    print(len(final_df))

# # Add 'event_id' and 'event_label' columns ----------------------------------------------
# filtered_data['event_id'] = np.arange(len(filtered_data))  # Sequential event identifiers
# filtered_data['event_label'] = 'date_filtered'  # Label for the events

# # Reorder columns to place 'event_id' and 'event_label' as the first columns
# columns_to_move = ['event_id', 'event_label']
# remaining_columns = [col for col in filtered_data.columns if col not in columns_to_move]
# filtered_data = filtered_data[columns_to_move + remaining_columns]

# # Save the DataFrame to a CSV file
# if debug_mode:
#     filtered_data.to_csv('hey.csv', sep=' ', index=False)

print("-------------------- Filter 1.1.1: uncalibrated data ---------------------")
# FILTER 2: TF, TB, QF, QB PRECALIBRATED THRESHOLDS --> 0 if out ------------------------------
for col in final_df.columns:
    if col.startswith('T') and col.endswith('_F'):
        final_df[col] = np.where((final_df[col] > T_F_right_pre_cal) | (final_df[col] < T_F_left_pre_cal), 0, final_df[col])
    if col.startswith('T') and col.endswith('_B'):
        final_df[col] = np.where((final_df[col] > T_B_right_pre_cal) | (final_df[col] < T_B_left_pre_cal), 0, final_df[col])
    if col.startswith('Q') and col.endswith('_F'):
        final_df[col] = np.where((final_df[col] > Q_F_right_pre_cal) | (final_df[col] < Q_F_left_pre_cal), 0, final_df[col])
    if col.startswith('Q') and col.endswith('_B'):
        final_df[col] = np.where((final_df[col] > Q_B_right_pre_cal) | (final_df[col] < Q_B_left_pre_cal), 0, final_df[col])


# New channel-wise plot -------------------------------------------------------
log_scale = True
if debug_mode:
    T_clip_min = -500
    T_clip_max = 500
    Q_clip_min = -500
    Q_clip_max = 500
    num_bins = 100  # Parameter for the number of bins
else:
    T_clip_min = -300
    T_clip_max = 100
    Q_clip_min = 0
    Q_clip_max = 500
    num_bins = 100  # Parameter for the number of bins


if create_plots or create_essential_plots:
    # Create the grand figure for T values
    fig_T, axes_T = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_T = axes_T.flatten()
    
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key}_F_{j+1}'
            col_B = f'{key}_B_{j+1}'
            y_F = final_df[col_F]
            y_B = final_df[col_B]
            
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
            y_F = final_df[col_F]
            y_B = final_df[col_B]
            
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Comprobation of emptiness of the columns
# -----------------------------------------------------------------------------

# Count the number of nonzero values in each column
nonzero_counts = (final_df != 0).sum()

# Identify columns with fewer than 100 nonzero values
low_value_cols = nonzero_counts[nonzero_counts < 100].index.tolist()

if low_value_cols:
    print(f"Warning: The following columns contain fewer than 100 nonzero values and may require review: {low_value_cols}")
    print("Rejecting file due to insufficient data.")

    # Move the file to the error directory
    final_path = os.path.join(base_directories["error_directory"], file_name)
    print(f"Moving {file_path} to the error directory {final_path}...")
    shutil.move(file_path, final_path)
    
    sys.exit()

# -----------------------------------------------------------------------------

charge_test = final_df.copy()
charge_test_copy = charge_test.copy()

# New pedestal calibration for charges ------------------------------------------------
QF_pedestal = []
for key in ['1', '2', '3', '4']:
    Q_F_cols = [f'Q{key}_F_{i+1}' for i in range(4)]
    Q_F = final_df[Q_F_cols].values
    
    Q_B_cols = [f'Q{key}_B_{i+1}' for i in range(4)]
    Q_B = final_df[Q_B_cols].values
    
    T_F_cols = [f'T{key}_F_{i+1}' for i in range(4)]
    T_F = final_df[T_F_cols].values
    
    QF_pedestal_component = [calibrate_strip_Q_pedestal(Q_F[:,i], T_F[:,i], Q_B[:,i]) for i in range(4)]
    QF_pedestal.append(QF_pedestal_component)
QF_pedestal = np.array(QF_pedestal)

QB_pedestal = []
for key in ['1', '2', '3', '4']:
    Q_F_cols = [f'Q{key}_F_{i+1}' for i in range(4)]
    Q_F = final_df[Q_F_cols].values
    
    Q_B_cols = [f'Q{key}_B_{i+1}' for i in range(4)]
    Q_B = final_df[Q_B_cols].values
    
    T_B_cols = [f'T{key}_B_{i+1}' for i in range(4)]
    T_B = final_df[T_B_cols].values
    
    QB_pedestal_component = [calibrate_strip_Q_pedestal(Q_B[:,i], T_B[:,i], Q_F[:,i]) for i in range(4)]
    QB_pedestal.append(QB_pedestal_component)
QB_pedestal = np.array(QB_pedestal)

print("\nFront Charge Pedestal:")
print(QF_pedestal)
print("\nBack Charge Pedestal:")
print(QB_pedestal)

for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    for j in range(4):
        mask = charge_test_copy[f'{key}_F_{j+1}'] != 0
        charge_test.loc[mask, f'{key}_F_{j+1}'] -= QF_pedestal[i][j]

for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    for j in range(4):
        mask = charge_test_copy[f'{key}_B_{j+1}'] != 0
        charge_test.loc[mask, f'{key}_B_{j+1}'] -= QB_pedestal[i][j]


# Plot histograms of all the pedestal substractions

if create_plots or create_essential_plots:

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
            
            Q_clip_min = -5
            Q_clip_max = 5
            
            # Plot histograms with Q-specific clipping and bins
            axes_Q[i*4 + j].hist(y_F[(y_F != 0) & (y_F > Q_clip_min) & (y_F < Q_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_F} (F)')
            axes_Q[i*4 + j].hist(y_B[(y_B != 0) & (y_B > Q_clip_min) & (y_B < Q_clip_max)], 
                                 bins=num_bins, alpha=0.5, label=f'{col_B} (B)')
            axes_Q[i*4 + j].set_title(f'{col_F} vs {col_B}')
            axes_Q[i*4 + j].legend()
            # Show between -5 and 5
            axes_Q[i*4 + j].set_xlim([-5, 5])
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
# -----------------------------------------------------------------------------



# --- Define FEE Calibration ---
FEE_calibration = {
    "Width": [
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
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
FEE_calibration = pd.DataFrame(FEE_calibration)
cs = CubicSpline(FEE_calibration['Width'].to_numpy(),
                 FEE_calibration['Fast Charge'].to_numpy(),
                 bc_type='natural')

def interpolate_fast_charge(width_array):
    """ Interpolates fast charge for array-like width values using cubic spline. """
    width_array = np.asarray(width_array)
    return np.where(width_array == 0, 0, cs(width_array))

# --- Calibrate and store new columns in final_df ---
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


import matplotlib.pyplot as plt

Q_clip_min = 0
Q_clip_max = 1750
num_bins = 100
log_scale = True

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

# --- Save/Show ---
if save_plots:
    final_filename = f'{fig_idx}_grand_figure_Q_fC.png'
    fig_idx += 1
    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
    plot_list.append(save_fig_path)
    plt.savefig(save_fig_path, format='png')

if show_plots:
    plt.show()
plt.close(fig_Q)


# -----------------------------------------------------------------------

pos_test = final_df.copy()

for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
    for j in range(4):
        pos_test[f'{key}_diff_{j+1}'] = ( pos_test[f'{key}_F_{j+1}'] - pos_test[f'{key}_B_{j+1}'] ) / 2

# print('Check this out:')
# print(pos_test[f'P1_diff_1'])
# print('---------')

pos_test_copy = pos_test.copy()

# New calibration for positions ------------------------------------------------
Tdiff_cal = []
for key in ['1', '2', '3', '4']:
    T_F_cols = [f'T{key}_F_{i+1}' for i in range(4)]
    T_F = final_df[T_F_cols].values
    
    T_B_cols = [f'T{key}_B_{i+1}' for i in range(4)]
    T_B = final_df[T_B_cols].values
    
    Tdiff_cal_component = [calibrate_strip_T_diff(T_F[:,i], T_B[:,i]) for i in range(4)]
    Tdiff_cal.append(Tdiff_cal_component)
Tdiff_cal = np.array(Tdiff_cal)

print("\nTime diff. offset:")
print(Tdiff_cal)

for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
    for j in range(4):
        mask = pos_test_copy[f'{key}_diff_{j+1}'] != 0
        pos_test.loc[mask, f'{key}_diff_{j+1}'] -= Tdiff_cal[i][j]


if create_plots:
    # Create the grand figure for Q values
    fig_Q, axes_Q = plt.subplots(4, 4, figsize=(20, 10))  # Adjust the layout as necessary
    axes_Q = axes_Q.flatten()
    
    for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
        for j in range(4):
            col_F = f'{key}_diff_{j+1}'
            y_F = pos_test[col_F]
            
            Q_clip_min = -5
            Q_clip_max = 5
            
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




# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# Compute T_sum, T_diff, Q_sum, Q_diff ----------------------------------------
new_columns_data = {'datetime': final_df['datetime'].values}
for key in ['T1', 'T2', 'T3', 'T4']:
    T_F_cols = [f'{key}_F_{i+1}' for i in range(4)]
    T_B_cols = [f'{key}_B_{i+1}' for i in range(4)]
    Q_F_cols = [f'{key.replace("T", "Q")}_F_{i+1}' for i in range(4)]
    Q_B_cols = [f'{key.replace("T", "Q")}_B_{i+1}' for i in range(4)]

    T_F = final_df[T_F_cols].values
    T_B = final_df[T_B_cols].values
    Q_F = final_df[Q_F_cols].values
    Q_B = final_df[Q_B_cols].values

    for i in range(4):
        new_columns_data[f'{key}_T_sum_{i+1}'] = (T_F[:, i] + T_B[:, i]) / 2
        new_columns_data[f'{key}_T_diff_{i+1}'] = (T_F[:, i] - T_B[:, i]) / 2
        new_columns_data[f'{key.replace("T", "Q")}_Q_sum_{i+1}'] = (Q_F[:, i] + Q_B[:, i]) / 2
        new_columns_data[f'{key.replace("T", "Q")}_Q_diff_{i+1}'] = (Q_F[:, i] - Q_B[:, i]) / 2

new_df = pd.DataFrame(new_columns_data)
timestamp_column = new_df['datetime']  # Adjust if the column name is different
data_columns = new_df.drop(columns=['datetime'])


print("----------------------- Filter 1.1: extreme outliers -----------------------")
data_columns = data_columns.applymap(lambda x: 0 if builtins.isinstance(x, (builtins.int, builtins.float)) and (x < -1e6 or x > 1e6) else x)
new_df = pd.concat([timestamp_column, data_columns], axis=1)

if debug_mode:
    print(len(new_df))

if create_plots:

    num_columns = len(new_df.columns) - 1  # Exclude 'datetime'
    num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
    fig, axes = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2))
    axes = axes.flatten()

    for i, col in enumerate([col for col in new_df.columns if col != 'datetime']):
        y = new_df[col]
        
        if 'Q_sum' in col:
            color = Q_sum_color
        if 'Q_diff' in col:
            color = Q_diff_color
        if 'T_sum' in col:
            color = T_sum_color
        if 'T_diff' in col:
            color = T_diff_color
        axes[i].hist(y[y != 0], bins=100, alpha=0.5, label=col, color=color)
        
        axes[i].set_title(col)
        axes[i].legend()
        if 'Q_sum' in col:
            axes[i].set_yscale('log')
    
    # Remove any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.suptitle("Uncalibrated data")
    
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



print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("--------------------- Filters and calibrations -----------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

print("-------------------- Filter 2: uncalibrated data ---------------------")

# FILTER 2: TSUM, TDIF, QSUM, QDIF PRECALIBRATED THRESHOLDS --> 0 if out ------------------------------
for col in new_df.columns:
    if 'T_sum' in col:
        new_df[col] = np.where((new_df[col] > T_sum_right_pre_cal) | (new_df[col] < T_sum_left_pre_cal), 0, new_df[col])
    if 'T_diff' in col:
        new_df[col] = np.where((new_df[col] > T_diff_pre_cal_threshold) | (new_df[col] < -T_diff_pre_cal_threshold), 0, new_df[col])
    if 'Q_sum' in col:
        new_df[col] = np.where((new_df[col] > Q_right_pre_cal) | (new_df[col] < Q_left_pre_cal), 0, new_df[col])
    if 'Q_diff' in col:
        new_df[col] = np.where((new_df[col] > Q_diff_pre_cal_threshold) | (new_df[col] < -Q_diff_pre_cal_threshold), 0, new_df[col])

calibrated_data = new_df.copy()

if debug_mode:
    print(len(calibrated_data))

print("----------------------------------------------------------------------")
print("----------- Charge sum pedestal, calibration and filtering -----------")
print("----------------------------------------------------------------------")

if debug_mode:
    print(calibrated_data)

for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    for j in range(4):
        mask = new_df[f'{key}_Q_sum_{j+1}'] != 0
        # calibrated_data.loc[mask, f'{key}_Q_sum_{j+1}'] -= calibration_Q[i][j]
        calibrated_data.loc[mask, f'{key}_Q_sum_{j+1}'] -= ( QF_pedestal[i][j] + QB_pedestal[i][j] ) / 2


print("--------------------- Filter 3: calibrated data ----------------------")
for col in calibrated_data.columns:
    if 'Q_sum' in col:
        calibrated_data[col] = np.where((calibrated_data[col] > Q_sum_right_cal) | (calibrated_data[col] < Q_sum_left_cal), 0, calibrated_data[col])


print("--------------------- Filter 3.1: if one charge is 0 put the time to 0 ----------------------")
for key in ['T1', 'T2', 'T3', 'T4']:
    for i in range(4):
        mask = (calibrated_data[f'{key.replace("T", "Q")}_Q_sum_{i+1}'] == 0)
        calibrated_data.loc[mask, [f'{key}_T_diff_{i+1}', f'{key}_T_sum_{i+1}', 
                                   f'{key.replace("T", "Q")}_Q_diff_{i+1}', f'{key.replace("T", "Q")}_Q_sum_{i+1}']] = 0


print("----------------------------------------------------------------------")
print("----------------- Time diff calibration and filtering ----------------")
print("----------------------------------------------------------------------")

calibration_T = []
for key in ['T1', 'T2', 'T3', 'T4']:
    T_dif_cols = [f'{key}_T_diff_{i+1}' for i in range(4)]
    T_dif = new_df[T_dif_cols].values
    calibration_t_component = [calibrate_strip_T(T_dif[:, i]) for i in range(4)]
    calibration_T.append(calibration_t_component)
calibration_T = np.array(calibration_T)

print(f"Time dif calibration:\n{calibration_T}")

diff = np.abs(calibration_T - time_dif_reference) > time_dif_distance
nan_mask = np.isnan(calibration_T)
values_replaced_t_dif = np.any(diff | nan_mask)
calibration_T[diff | nan_mask] = time_dif_reference[diff | nan_mask]
if values_replaced_t_dif:
    print("Some values were replaced in the calibration T dif.")

for i, key in enumerate(['T1', 'T2', 'T3', 'T4']):
    for j in range(4):
        mask = new_df[f'{key}_T_diff_{j+1}'] != 0
        # calibrated_data.loc[mask, f'{key}_T_diff_{j+1}'] -= calibration_T[i][j]
        calibrated_data.loc[mask, f'{key}_T_diff_{j+1}'] -= Tdiff_cal[i][j]

print("--------------------- Filter 3.2: time diff filtering ----------------------")
for col in calibrated_data.columns:
    if 'T_diff' in col:
        calibrated_data[col] = np.where((calibrated_data[col] > T_diff_cal_threshold) | (calibrated_data[col] < -T_diff_cal_threshold), 0, calibrated_data[col])

for key in ['T1', 'T2', 'T3', 'T4']:
    for i in range(4):
        mask = (calibrated_data[f'{key}_T_diff_{i+1}'] == 0)
        calibrated_data.loc[mask, [f'{key}_T_diff_{i+1}', f'{key}_T_sum_{i+1}', 
                                   f'{key.replace("T", "Q")}_Q_diff_{i+1}', f'{key.replace("T", "Q")}_Q_sum_{i+1}']] = 0


print("----------------------------------------------------------------------")
print("---------------- Charge diff calibration and filtering ---------------")
print("----------------------------------------------------------------------")

for i, key in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
    for j in range(4):
        mask = new_df[f'{key}_Q_diff_{j+1}'] != 0
        # calibrated_data.loc[mask, f'{key}_Q_diff_{j+1}'] -= calibration_Q_FB[i][j]
        calibrated_data.loc[mask, f'{key}_Q_diff_{j+1}'] -= ( QF_pedestal[i][j] - QB_pedestal[i][j] ) / 2

# Add datetime column to calibrated_data -----------------------------
calibrated_data['datetime'] = final_df['datetime']


print("------------------ Filter 4: charge diff filtering -------------------")
for col in calibrated_data.columns:
    if 'Q_diff' in col:
        calibrated_data[col] = np.where((calibrated_data[col] > Q_diff_cal_threshold) | (calibrated_data[col] < -Q_diff_cal_threshold), 0, calibrated_data[col])


print("------------------ Filter 4.1: one-side filter removal -------------------")
for key in ['T1', 'T2', 'T3', 'T4']:
    for i in range(4):
        mask = (calibrated_data[f'{key}_T_diff_{i+1}'] == 0) | (calibrated_data[f'{key}_T_sum_{i+1}'] == 0) | \
               (calibrated_data[f'{key.replace("T", "Q")}_Q_diff_{i+1}'] == 0) | (calibrated_data[f'{key.replace("T", "Q")}_Q_sum_{i+1}'] == 0)
        calibrated_data.loc[mask, [f'{key}_T_diff_{i+1}', f'{key}_T_sum_{i+1}', 
                                   f'{key.replace("T", "Q")}_Q_diff_{i+1}', f'{key.replace("T", "Q")}_Q_sum_{i+1}']] = 0


if create_plots:
    num_columns = len(calibrated_data.columns) - 1  # Exclude 'datetime'
    num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
    fig, axes = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2))
    axes = axes.flatten()
    for i, col in enumerate([col for col in calibrated_data.columns if col != 'datetime']):
        y = calibrated_data[col]
        
        if 'Q_sum' in col:
            color = Q_sum_color
        if 'Q_diff' in col:
            color = Q_diff_color
        if 'T_sum' in col:
            color = T_sum_color
        if 'T_diff' in col:
            color = T_diff_color
        axes[i].hist(y[y != 0], bins=300, alpha=0.5, label=col, color=color)
        
        axes[i].set_title(col)
        axes[i].legend()
        if 'Q_sum' in col:
            axes[i].set_yscale('log') 
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.suptitle('Calibrated data, one-side events removed')
    
    if save_plots:
        name_of_file = 'calibrated_one_side_removed'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: plt.show()
    plt.close()


# For articles and presentations
if presentation_plots:
    plane = 2
    strip = 2
    data = [f'P{plane}_T_sum_{strip}', f'P{plane}_T_diff_{strip}', f'Q{plane}_Q_sum_{strip}', f'Q{plane}_Q_diff_{strip}']
    fig_idx = 0  # Assuming fig_idx is defined earlier
    plot_list = []  # Assuming plot_list is defined earlier

    for i, col in enumerate([col for col in calibrated_data.columns if col != 'datetime'][:len(data)]):
        y = calibrated_data[col]
        
        if 'Q_sum' in col:
            color = 'green'
        elif 'Q_diff' in col:
            color = 'blue'
        elif 'T_sum' in col:
            color = T_sum_color
        elif 'T_diff' in col:
            color = 'red'
        
        y_p = y[y != 0]
        
        # Create a new figure for each histogram
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(y_p, bins=500, label=f"{len(y_p)} entries", alpha=0.5, color=color)
        
        if 'T_diff' in col:
            ax.set_xlim([-1, 1])
        if 'Q_diff' in col:
            ax.set_xlim([-2, 2])
        if 'Q_sum' in col:
            ax.set_yscale('log')
            ax.set_xlim([-5, 50])
            
        if 'Q' in col:
            ax.set_xlabel('QtW / ns')
        elif 'T' in col:
            ax.set_xlabel('T / ns')
        ax.set_ylabel('Counts')
        
        # ax.set_title(data[i])
        ax.legend(frameon=False, handletextpad=0, handlelength=0)
        plt.tight_layout()
        
        if save_plots:
            name_of_file = data[i].replace(' ', '_').replace('/', '_')  # Sanitize file name
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1

            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
            # plt.savefig(final_filename, format='png', dpi=300)
        
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
            Q_sum = calibrated_data[f'Q{key}_Q_sum_{i+1}'].values
            Q_diff = calibrated_data[f'Q{key}_Q_diff_{i+1}'].values

            # Apply condition to filter non-zero Q_sum and Q_diff
            cond = (Q_sum != 0) & (Q_diff != 0)
            Q_sum_adjusted = Q_sum[cond].copy()
            Q_diff_adjusted = Q_diff[cond].copy()
            
            # print(len(Q_sum_adjusted))
            # print(len(Q_diff_adjusted))
            
            # Skip correction if no data is left after filtering
            if np.sum(Q_sum_adjusted) == 0:
                continue

            # Perform scatter plot and fit
            title = f"Q{key}_{i+1}. Charge diff. vs. charge sum."
            x_label = "Charge sum"
            y_label = "Charge diff"
            name_of_file = f"Q{key}_{i+1}_charge_analysis_scatter_diff_vs_sum"
            coeffs = scatter_2d_and_fit_new(Q_sum_adjusted, Q_diff_adjusted, title, x_label, y_label, name_of_file)
            
            # Calculate the correction based on filtered data
            # correction = polynomial(Q_diff[Q_diff != 0], *coeffs)

            # # Apply correction directly to non-zero values
            # Q_diff[Q_diff != 0] -= correction

            # # Update the DataFrame with corrected values
            # calibrated_data[f'Q{key}_Q_diff_{i+1}'] = Q_diff
            
            # Update only filtered rows in the DataFrame
            calibrated_data.loc[cond, f'Q{key}_Q_diff_{i+1}'] = Q_diff_adjusted - polynomial(Q_sum_adjusted, *coeffs)

        
    # ADD THE SUBFIGURES HERE OF THE 16 CALIBRATIONS
    print('Charge front-back correction performed.')
    
    if create_plots or create_essential_plots:
        num_columns = len(calibrated_data.columns) - 1  # Exclude 'datetime'
        num_rows = (num_columns + 7) // 8  # Adjust as necessary for better layout
        fig, axes = plt.subplots(num_rows, 8, figsize=(20, num_rows * 2))
        axes = axes.flatten()
        for i, col in enumerate([col for col in calibrated_data.columns if col != 'datetime']):
            y = calibrated_data[col]
            axes[i].hist(y[y != 0], bins=300, alpha=0.5, label=col)
            axes[i].set_title(col)
            axes[i].legend()
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.suptitle('Calibrated data, included Front-back correction')
        
        if save_plots:
            name_of_file = 'calibrated_including_ch_diff'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        
        if show_plots:
            plt.show()
        plt.close()

else:
    print('Charge front-back correction was selected to not be performed.')
    Q_diff_cal_threshold_FB = 10


print("----------------- Filter 5: charge difference FB filter -----------------")
for col in calibrated_data.columns:
    if 'Q_diff' in col:
        calibrated_data[col] = np.where((calibrated_data[col] > Q_diff_cal_threshold_FB) | (calibrated_data[col] < Q_diff_cal_threshold_FB), 0, calibrated_data[col])

print("----------------------------------------------------------------------")
print("------------------------ Slewing correction --------------------------")
print("----------------------------------------------------------------------")

if slewing_correction:
    print("WIP")
    
    data_df = calibrated_data.copy()
    
    # Take only the columns which have T_sum in the name, I mean, inside of the column name
    # Also _final cannot be in the column name. Also the column called 'type' save it
    # Extract relevant columns
    data_df_times = data_df.filter(regex='T_sum')
    data_df_charges = data_df.filter(regex='Q_sum')
    data_df_tdiff = data_df.filter(regex='T_diff')

    # Concatenate all relevant data with 'type' column
    data_df_times = pd.concat([data_df_charges, data_df_times, data_df_tdiff], axis=1)
    print(data_df_times.columns.to_list())
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

    # Print the resulting z_positions
    z_positions = z_positions - z_positions[0]
    print(f"Z positions: {z_positions}")

    def y_pos(y_width):
        """Returns array of y-centers based on the widths of each strip."""
        return np.cumsum(y_width) - (np.sum(y_width) + y_width) / 2

    # For P1/P3 vs P2/P4
    y_widths = [
        np.array([63, 63, 63, 98]),  # P1 / P3
        np.array([98, 63, 63, 63])   # P2 / P4
    ]

    y_pos_P1_P3 = y_pos(y_widths[0])  # shape (4,)
    y_pos_P2_P4 = y_pos(y_widths[1])  # shape (4,)

    # yz_big[plane_index, strip_index, 0/1] = (y,z)
    # plane_index in [0..3] => P1=0, P2=1, P3=2, P4=3
    # strip_index in [0..3], but your data columns say 1..4 => we do minus 1
    yz_big = np.zeros((4, 4, 2))

    # Fill P1 -> plane_idx=0, P3 -> plane_idx=2 with y_pos_P1_P3
    for strip_idx in range(4):
        yz_big[0, strip_idx, 0] = y_pos_P1_P3[strip_idx]  # y
        yz_big[0, strip_idx, 1] = z_positions[0]         # z for P1
        
        yz_big[2, strip_idx, 0] = y_pos_P1_P3[strip_idx]  # y
        yz_big[2, strip_idx, 1] = z_positions[2]         # z for P3

    # Fill P2 -> plane_idx=1, P4 -> plane_idx=3 with y_pos_P2_P4
    for strip_idx in range(4):
        yz_big[1, strip_idx, 0] = y_pos_P2_P4[strip_idx]  # y
        yz_big[1, strip_idx, 1] = z_positions[1]         # z for P2

        yz_big[3, strip_idx, 0] = y_pos_P2_P4[strip_idx]  # y
        yz_big[3, strip_idx, 1] = z_positions[3]         # z for P4

    # Mapping P1->0, P2->1, P3->2, P4->3
    plane_map = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}

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
    #    (e.g. "P1_T_sum_3" => plane="P1", strip=3)
    ##############################################################################
    def parse_plane_and_strip(col_name):
        """
        This regex expects columns like "P1_T_sum_2" or "P3_T_diff_4".
        If the column doesn't match, raises ValueError.
        """
        pattern = r"^(T[1-4])_T_(?:sum|diff)_(\d+)$"
        match = re.match(pattern, col_name)
        if not match:
            raise ValueError(f"Cannot parse plane/strip from '{col_name}'")
        plane_str = match.group(1)  # e.g. "P1"
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
    # ['type', 'Q1_Q_sum_1', 'Q1_Q_sum_2', ..., 'P1_T_sum_1', ..., 'P2_T_diff_2', ...]

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

    import matplotlib.pyplot as plt

    # Get available T_sum and Q_sum difference columns
    num_available_T = len(T_sum_diff_df.columns)
    num_available_Q = len(Q_sum_diff_df.columns)

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
            
            if create_essential_plots or create_plots:
                # Plot in a scatter plot the original, the fit and the corrected values
                plt.figure(figsize=(6, 4))
                plt.scatter(y, x, alpha=0.6, s=1, label="Original")
                plt.scatter(y, slope * y + intercept, alpha=0.6, s=1, label="Fit")
                plt.scatter(y, corrected_x, alpha=0.6, s=1, label="Corrected")
                plt.xlabel(col_q)
                plt.ylabel(col_t)
                plt.xlim([-100, 100])
                plt.ylim([-4, 4])
                plt.title(f"{col_t} vs. {col_q}")
                plt.legend()
                
                if save_plots:
                    name_of_file = 'slew_corr'
                    final_filename = f'{fig_idx}_{name_of_file}.png'
                    fig_idx += 1
                    
                    save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
                    plot_list.append(save_fig_path)
                    plt.savefig(save_fig_path, format='png')
                
                if show_plots:
                    plt.show()
                plt.close()

    else:
        print("No valid combination of T_sum/Q_sum differences to plot.")


    from scipy.stats import linregress
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    import re

    # -----------------------------
    # Step 1: Per-strip slewing correction
    # -----------------------------
    corrected_T_sum = {}

    for col_t in T_sum_cols:
        match = re.match(r"^(T[1-4])_T_sum_(\d+)$", col_t)
        if not match:
            continue
        plane = match.group(1)
        strip = match.group(2)
        col_q = f"{plane.replace('T', 'Q')}_Q_sum_{strip}"

        if col_q not in data_case.columns:
            print(f"Missing Q_sum for {col_t} → skipping.")
            continue

        x = data_case[col_q]
        y = data_case[col_t]

        cond = (x != 0) & (y != 0) & np.isfinite(x) & np.isfinite(y)
        x_fit = x[cond]
        y_fit = y[cond]

        if len(x_fit) < 100:
            print(f"Insufficient points for {col_t} → skipping correction.")
            corrected_T_sum[col_t] = data_case[col_t]
            continue

        try:
            slope, intercept, _, _, _ = linregress(x_fit, y_fit)
            corrected_y = data_case[col_t] - (slope * data_case[col_q] + intercept)
            corrected_T_sum[col_t] = corrected_y
        except Exception as e:
            print(f"Fit failed for {col_t}: {e}")
            corrected_T_sum[col_t] = data_case[col_t]

    corrected_T_sum_df = pd.DataFrame(corrected_T_sum)

    # -----------------------------
    # Step 2: Compute corrected T_sum differences
    # -----------------------------
    corrected_T_sum_diff_df = {
        f"{c1}-{c2}": corrected_T_sum_df[c1] - corrected_T_sum_df[c2]
        for c1, c2 in itertools.combinations(corrected_T_sum_df.columns, 2)
    }
    corrected_T_sum_diff_df = pd.DataFrame(corrected_T_sum_diff_df)

    # -----------------------------
    # Step 3: Identify valid inter-layer combinations
    # -----------------------------
    valid_pairs = []
    pattern = r"^(T[1-4])_T_sum_(\d+)-(T[1-4])_T_sum_(\d+)$"

    for col_t in corrected_T_sum_diff_df.columns:
        m = re.match(pattern, col_t)
        if not m:
            continue
        plane1, _, plane2, _ = m.groups()
        if plane1 != plane2:
            col_q = col_t.replace("T_sum", "Q_sum").replace("T", "Q")
            if col_q in Q_sum_diff_df.columns:
                valid_pairs.append((col_t, col_q))

    # -----------------------------
    # Step 4: Plot first 5 valid inter-layer pairs
    # -----------------------------
    num_plots = min(5, len(valid_pairs))
    if num_plots > 0:
        fig, axes = plt.subplots(num_plots, 1, figsize=(6, 4 * num_plots))
        if num_plots == 1:
            axes = [axes]

        for i in range(num_plots):
            col_t, col_q = valid_pairs[i]

            x = corrected_T_sum_diff_df[col_t]
            y = Q_sum_diff_df[col_q]

            cond = (abs(x) <= 5) & (abs(y) <= 100) & np.isfinite(x) & np.isfinite(y)
            x = x[cond]
            y = y[cond]

            axes[i].scatter(y, x, alpha=0.6, s=1)
            axes[i].set_xlabel(col_q)
            axes[i].set_ylabel(col_t)
            axes[i].set_title(f"{col_t} vs {col_q} (after per-strip slewing correction)")
            axes[i].set_xlim(-100, 100)
            axes[i].set_ylim(-4, 4)  

        plt.tight_layout()

        if save_plots:
            name_of_file = 'slew_corr_first5_interlayer_scatter'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')

        if show_plots:
            plt.show()
        plt.close()
    else:
        print("No valid inter-layer T_sum/Q_sum pairs found for plotting.")


print("----------------------------------------------------------------------")
print("----------------------- Time sum calibration -------------------------")
print("----------------------------------------------------------------------")

if time_calibration:
    # Initialize an empty list to store the resulting matrices for each event
    event_matrices = []
    
    # Iterate over each event (row) in the DataFrame
    for _, row in calibrated_data.iterrows():
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
    P1s1_P4s1 = []
    P1s1_P4s2 = []
    P1s2_P4s1 = []
    P1s2_P4s2 = []
    P1s2_P4s3 = []
    P1s3_P4s2 = []
    P1s3_P4s3 = []
    P1s3_P4s4 = []
    P1s4_P4s3 = []
    P1s4_P4s4 = []
    P1s1_P4s3 = []
    P1s3_P4s1 = []
    P1s2_P4s4 = []
    P1s4_P4s2 = []
    P1s1_P4s4 = []
    
    # Two layers spaced
    P1s1_P3s1 = []
    P1s1_P3s2 = []
    P1s2_P3s1 = []
    P1s2_P3s2 = []
    P1s2_P3s3 = []
    P1s3_P3s2 = []
    P1s3_P3s3 = []
    P1s3_P3s4 = []
    P1s4_P3s3 = []
    P1s4_P3s4 = []
    P1s1_P3s3 = []
    P1s3_P3s1 = []
    P1s2_P3s4 = []
    P1s4_P3s2 = []
    P1s1_P3s4 = []
    
    P2s1_P4s1 = []
    P2s1_P4s2 = []
    P2s2_P4s1 = []
    P2s2_P4s2 = []
    P2s2_P4s3 = []
    P2s3_P4s2 = []
    P2s3_P4s3 = []
    P2s3_P4s4 = []
    P2s4_P4s3 = []
    P2s4_P4s4 = []
    P2s1_P4s3 = []
    P2s3_P4s1 = []
    P2s2_P4s4 = []
    P2s4_P4s2 = []
    P2s1_P4s4 = []
    
    # One layer spaced
    P1s1_P2s1 = []
    P1s1_P2s2 = []
    P1s2_P2s1 = []
    P1s2_P2s2 = []
    P1s2_P2s3 = []
    P1s3_P2s2 = []
    P1s3_P2s3 = []
    P1s3_P2s4 = []
    P1s4_P2s3 = []
    P1s4_P2s4 = []
    P1s1_P2s3 = []
    P1s3_P2s1 = []
    P1s2_P2s4 = []
    P1s4_P2s2 = []
    P1s1_P2s4 = []
    
    P2s1_P3s1 = []
    P2s1_P3s2 = []
    P2s2_P3s1 = []
    P2s2_P3s2 = []
    P2s2_P3s3 = []
    P2s3_P3s2 = []
    P2s3_P3s3 = []
    P2s3_P3s4 = []
    P2s4_P3s3 = []
    P2s4_P3s4 = []
    P2s1_P3s3 = []
    P2s3_P3s1 = []
    P2s2_P3s4 = []
    P2s4_P3s2 = []
    P2s1_P3s4 = []
    
    P3s1_P4s1 = []
    P3s1_P4s2 = []
    P3s2_P4s1 = []
    P3s2_P4s2 = []
    P3s2_P4s3 = []
    P3s3_P4s2 = []
    P3s3_P4s3 = []
    P3s3_P4s4 = []
    P3s4_P4s3 = []
    P3s4_P4s4 = []
    P3s1_P4s3 = []
    P3s3_P4s1 = []
    P3s2_P4s4 = []
    P3s4_P4s2 = []
    P3s1_P4s4 = []
    
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
        # Convert data to numpy arrays and filter
        pos_x = np.array(pos_x)
        pos_x = pos_x[(-200 < pos_x) & (pos_x < 200) & (pos_x != 0)]
        v_travel_time = np.array(v_travel_time)
        v_travel_time = v_travel_time[v_travel_time < 1.6]
        t_0 = np.array(t_0)
        t_0 = t_0[(-10 < t_0) & (t_0 < 10)]
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
    
        # No fit: loop over each vector and plot histogram
        for i, vector in enumerate(vectors):
            var_name = [name for name, val in globals().items() if val is vector][0]
            if i >= number_of_time_cal_figures: break
            hist_1d(vector, 100, var_name, "T / ns", var_name)


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
    
    crt_df = pd.DataFrame(crt_values, index=calibrated_data.index)
    calibrated_data = pd.concat([calibrated_data, crt_df], axis=1)
    calibrated_data = calibrated_data.copy()
    crt_values = calibrated_data.filter(like='CRT_').iloc[0].values
    Q1, Q3 = np.percentile(crt_values, [25, 75])
    crt_values = crt_values[crt_values <= 1]
    filtered_crt_values = crt_values[(crt_values >= Q1 - 1.5 * (Q3 - Q1)) & (crt_values <= Q3 + 1.5 * (Q3 - Q1))]
    
    global_variables['CRP_avg'] = np.mean(filtered_crt_values)*1000
    
    # print(f"CRT values: {crt_values}, Filtered: {filtered_crt_values}, Avg: {calibrated_data['CRP_avg'][0]:.4g}")
    print("---------------------------")
    print(f"CRT Avg: {global_variables['CRP_avg']:.4g} ps")
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
    
    # -----------------------------------------------------------------------------
    # Brute force method
    # -----------------------------------------------------------------------------
    brute_force_analysis = False
    if brute_force_analysis:
        # Main itinerary
        itinerary = ["P1s1", "P3s1", "P1s2", "P3s2", "P1s3", "P3s3", "P1s4", "P3s4","P4s4", "P2s4", "P4s3", "P2s3", "P4s2", "P2s2", "P4s1", "P2s1"]
        import random
        k = 0
        max_iter = 2000000
        brute_force_list = []
        # Create row and column indices
        rows = ['P{}'.format(i) for i in range(1, 5)]
        columns = ['s{}'.format(i) for i in range(1,5)]
        brute_force_df = pd.DataFrame(0, index=rows, columns=columns)
        jump = False
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
    
    # -----------------------------------------------------------------------------
    # Selected paths method
    # -----------------------------------------------------------------------------
    itineraries = [
    ['P1s1', 'P3s1', 'P1s2', 'P3s2', 'P1s3', 'P3s3', 'P1s4', 'P3s4', 'P4s4', 'P2s4', 'P4s3', 'P2s3', 'P4s2', 'P2s2', 'P4s1', 'P2s1'],
    ['P3s4', 'P1s4', 'P2s4', 'P4s4', 'P2s2', 'P4s3', 'P2s3', 'P1s3', 'P3s3', 'P2s1', 'P4s2', 'P1s2', 'P3s2', 'P1s1', 'P4s1', 'P3s1'],
    ['P3s2', 'P1s2', 'P2s2', 'P4s1', 'P3s1', 'P1s1', 'P3s3', 'P4s2', 'P2s3', 'P1s3', 'P3s4', 'P2s4', 'P4s4', 'P1s4', 'P4s3', 'P2s1'],
    ['P2s4', 'P4s2', 'P1s4', 'P4s4', 'P2s3', 'P4s1', 'P1s3', 'P3s3', 'P1s2', 'P2s2', 'P3s2', 'P2s1', 'P3s1', 'P1s1', 'P4s3', 'P3s4'],
    ['P2s4', 'P4s4', 'P2s2', 'P1s2', 'P3s1', 'P1s1', 'P4s3', 'P2s3', 'P4s1', 'P1s3', 'P3s4', 'P1s4', 'P3s3', 'P2s1', 'P4s2', 'P3s2'],
    ['P3s1', 'P2s1', 'P1s2', 'P4s3', 'P1s3', 'P2s2', 'P3s3', 'P4s1', 'P3s2', 'P1s1', 'P4s4', 'P2s3', 'P3s4', 'P2s4', 'P4s2', 'P1s4'],
    ['P2s3', 'P4s4', 'P2s4', 'P4s2', 'P1s1', 'P3s2', 'P2s1', 'P3s1', 'P4s1', 'P1s3', 'P2s2', 'P1s2', 'P3s3', 'P1s4', 'P4s3', 'P3s4'],
    ['P2s4', 'P3s4', 'P4s2', 'P1s1', 'P2s1', 'P3s1', 'P1s2', 'P4s1', 'P1s3', 'P4s4', 'P2s2', 'P3s3', 'P1s4', 'P2s3', 'P4s3', 'P3s2'],
    ['P3s3', 'P1s2', 'P3s2', 'P2s1', 'P4s3', 'P2s3', 'P4s4', 'P3s4', 'P2s4', 'P1s4', 'P4s2', 'P2s2', 'P1s3', 'P4s1', 'P1s1', 'P3s1'],
    ['P2s4', 'P3s4', 'P1s4', 'P3s3', 'P4s1', 'P2s3', 'P4s2', 'P2s1', 'P3s2', 'P1s3', 'P4s3', 'P2s2', 'P1s2', 'P4s4', 'P1s1', 'P3s1'],
    ['P4s2', 'P3s2', 'P4s3', 'P1s3', 'P2s2', 'P4s1', 'P1s1', 'P2s1', 'P3s3', 'P1s4', 'P2s3', 'P3s4', 'P2s4', 'P4s4', 'P1s2', 'P3s1'],
    ['P1s3', 'P2s3', 'P3s4', 'P1s4', 'P4s4', 'P2s4', 'P4s3', 'P1s2', 'P3s1', 'P4s1', 'P2s1', 'P4s2', 'P3s2', 'P1s1', 'P3s3', 'P2s2'],
    ['P2s4', 'P4s3', 'P1s2', 'P2s1', 'P3s2', 'P2s2', 'P4s2', 'P3s3', 'P1s4', 'P2s3', 'P1s3', 'P3s4', 'P4s4', 'P1s1', 'P3s1', 'P4s1'],
    ['P2s2', 'P1s2', 'P4s1', 'P1s1', 'P3s1', 'P2s1', 'P3s3', 'P4s2', 'P2s4', 'P4s4', 'P1s4', 'P2s3', 'P3s4', 'P4s3', 'P1s3', 'P3s2'],
    ['P3s1', 'P2s1', 'P3s3', 'P2s2', 'P4s2', 'P2s4', 'P4s4', 'P1s2', 'P3s2', 'P1s3', 'P3s4', 'P1s4', 'P2s3', 'P4s1', 'P1s1', 'P4s3'],
    ['P4s2', 'P3s2', 'P2s2', 'P4s4', 'P3s3', 'P1s4', 'P2s3', 'P1s3', 'P3s4', 'P2s4', 'P4s3', 'P2s1', 'P1s2', 'P3s1', 'P4s1', 'P1s1'],
    ['P1s2', 'P3s3', 'P4s4', 'P1s1', 'P4s1', 'P3s1', 'P2s1', 'P3s2', 'P1s3', 'P3s4', 'P2s3', 'P4s3', 'P2s2', 'P4s2', 'P2s4', 'P1s4'],
    ['P3s3', 'P1s2', 'P4s2', 'P3s2', 'P1s3', 'P2s2', 'P4s1', 'P1s1', 'P3s1', 'P2s1', 'P4s3', 'P1s4', 'P2s4', 'P3s4', 'P4s4', 'P2s3'],
    ['P3s4', 'P1s3', 'P4s2', 'P2s4', 'P4s3', 'P3s2', 'P1s2', 'P3s3', 'P2s2', 'P4s1', 'P2s3', 'P1s4', 'P4s4', 'P2s1', 'P1s1', 'P3s1'],
    ['P2s1', 'P1s1', 'P3s1', 'P1s2', 'P3s3', 'P1s4', 'P2s3', 'P4s4', 'P3s4', 'P4s2', 'P2s4', 'P4s3', 'P1s3', 'P2s2', 'P4s1', 'P3s2'],
    ['P3s3', 'P2s2', 'P1s2', 'P4s4', 'P2s1', 'P3s2', 'P1s3', 'P3s4', 'P1s4', 'P2s3', 'P4s1', 'P3s1', 'P1s1', 'P4s3', 'P2s4', 'P4s2'],
    ['P3s2', 'P2s2', 'P4s2', 'P2s4', 'P1s4', 'P3s4', 'P1s3', 'P4s1', 'P1s2', 'P3s1', 'P1s1', 'P3s3', 'P4s4', 'P2s3', 'P4s3', 'P2s1'],
    ['P3s2', 'P1s2', 'P4s2', 'P1s1', 'P4s4', 'P2s3', 'P1s4', 'P3s3', 'P2s1', 'P3s1', 'P4s1', 'P2s2', 'P1s3', 'P3s4', 'P2s4', 'P4s3'],
    ['P3s2', 'P2s2', 'P3s3', 'P1s1', 'P4s2', 'P1s3', 'P4s3', 'P3s4', 'P2s4', 'P1s4', 'P2s3', 'P4s4', 'P1s2', 'P4s1', 'P3s1', 'P2s1'],
    ['P1s3', 'P3s4', 'P2s4', 'P1s4', 'P3s3', 'P1s2', 'P2s1', 'P4s4', 'P2s3', 'P4s1', 'P3s2', 'P4s2', 'P2s2', 'P4s3', 'P1s1', 'P3s1'],
    ['P2s1', 'P3s3', 'P1s4', 'P2s3', 'P3s4', 'P1s3', 'P4s2', 'P1s1', 'P3s1', 'P4s1', 'P2s2', 'P3s2', 'P1s2', 'P4s3', 'P2s4', 'P4s4'],
    ['P3s1', 'P4s1', 'P3s2', 'P1s1', 'P4s2', 'P2s4', 'P1s4', 'P2s3', 'P1s3', 'P3s3', 'P2s2', 'P1s2', 'P4s4', 'P3s4', 'P4s3', 'P2s1'],
    ['P1s3', 'P3s4', 'P2s4', 'P4s2', 'P1s4', 'P4s4', 'P3s3', 'P2s3', 'P4s3', 'P3s2', 'P4s1', 'P2s1', 'P1s1', 'P3s1', 'P1s2', 'P2s2'],
    ['P3s2', 'P2s2', 'P1s3', 'P4s3', 'P1s4', 'P2s3', 'P4s2', 'P1s1', 'P4s1', 'P3s1', 'P2s1', 'P1s2', 'P3s3', 'P4s4', 'P2s4', 'P3s4'],
    ['P2s3', 'P3s3', 'P1s1', 'P3s1', 'P1s2', 'P4s2', 'P2s1', 'P3s2', 'P4s1', 'P2s2', 'P4s4', 'P1s3', 'P3s4', 'P4s3', 'P1s4', 'P2s4'],
    ['P1s1', 'P3s1', 'P1s2', 'P4s1', 'P2s1', 'P3s2', 'P1s3', 'P2s3', 'P1s4', 'P4s4', 'P2s2', 'P4s3', 'P2s4', 'P3s4', 'P4s2', 'P3s3'],
    ['P1s3', 'P3s3', 'P1s4', 'P2s4', 'P3s4', 'P4s2', 'P2s3', 'P4s4', 'P1s2', 'P3s2', 'P2s2', 'P4s3', 'P2s1', 'P4s1', 'P3s1', 'P1s1'],
    ['P2s3', 'P3s4', 'P2s4', 'P4s4', 'P1s1', 'P4s1', 'P2s2', 'P4s2', 'P1s2', 'P3s1', 'P2s1', 'P3s2', 'P1s3', 'P3s3', 'P4s3', 'P1s4'],
    ['P2s4', 'P4s4', 'P1s2', 'P4s2', 'P2s3', 'P3s4', 'P1s4', 'P3s3', 'P1s3', 'P4s1', 'P2s1', 'P4s3', 'P2s2', 'P3s2', 'P1s1', 'P3s1'],
    ['P4s3', 'P2s1', 'P1s2', 'P2s2', 'P3s2', 'P1s1', 'P3s1', 'P4s1', 'P3s3', 'P4s2', 'P2s4', 'P1s4', 'P4s4', 'P2s3', 'P3s4', 'P1s3'],
    ['P2s2', 'P4s4', 'P2s4', 'P4s3', 'P2s3', 'P4s1', 'P2s1', 'P1s1', 'P3s1', 'P1s2', 'P3s2', 'P4s2', 'P1s3', 'P3s3', 'P1s4', 'P3s4'],
    ['P3s1', 'P4s1', 'P2s3', 'P4s3', 'P1s1', 'P2s1', 'P1s2', 'P2s2', 'P4s2', 'P2s4', 'P4s4', 'P3s4', 'P1s4', 'P3s3', 'P1s3', 'P3s2'],
    ['P4s2', 'P3s3', 'P2s1', 'P1s2', 'P4s4', 'P2s2', 'P4s3', 'P1s3', 'P3s4', 'P2s4', 'P1s4', 'P2s3', 'P4s1', 'P3s1', 'P1s1', 'P3s2'],
    ['P1s3', 'P3s4', 'P2s4', 'P4s2', 'P1s1', 'P3s1', 'P1s2', 'P2s2', 'P4s4', 'P2s3', 'P1s4', 'P3s3', 'P4s3', 'P3s2', 'P4s1', 'P2s1'],
    ['P3s2', 'P1s3', 'P4s2', 'P3s3', 'P2s3', 'P3s4', 'P2s4', 'P1s4', 'P4s4', 'P2s2', 'P4s1', 'P2s1', 'P3s1', 'P1s2', 'P4s3', 'P1s1'],
    ['P2s3', 'P4s4', 'P2s4', 'P1s4', 'P3s4', 'P1s3', 'P3s2', 'P2s2', 'P4s2', 'P2s1', 'P4s3', 'P3s3', 'P1s1', 'P3s1', 'P1s2', 'P4s1'],
    ['P4s1', 'P3s1', 'P1s1', 'P2s1', 'P4s4', 'P1s3', 'P2s3', 'P4s3', 'P2s2', 'P3s2', 'P1s2', 'P4s2', 'P3s3', 'P1s4', 'P3s4', 'P2s4'],
    ['P2s4', 'P4s3', 'P2s3', 'P4s1', 'P1s3', 'P2s2', 'P3s2', 'P4s2', 'P1s2', 'P3s1', 'P2s1', 'P3s3', 'P1s1', 'P4s4', 'P1s4', 'P3s4'],
    ['P1s4', 'P2s4', 'P4s3', 'P2s3', 'P3s3', 'P1s1', 'P3s2', 'P4s1', 'P1s3', 'P3s4', 'P4s4', 'P2s2', 'P4s2', 'P2s1', 'P3s1', 'P1s2'],
    ['P2s2', 'P4s1', 'P2s3', 'P1s3', 'P3s2', 'P1s1', 'P3s1', 'P1s2', 'P3s3', 'P2s1', 'P4s3', 'P2s4', 'P3s4', 'P4s2', 'P1s4', 'P4s4'],
    ['P2s2', 'P1s2', 'P2s1', 'P3s2', 'P1s1', 'P4s3', 'P2s4', 'P4s2', 'P2s3', 'P3s4', 'P1s4', 'P3s3', 'P4s4', 'P1s3', 'P4s1', 'P3s1'],
    ['P2s1', 'P3s1', 'P4s1', 'P2s3', 'P3s3', 'P2s2', 'P3s2', 'P1s3', 'P4s4', 'P1s2', 'P4s2', 'P1s1', 'P4s3', 'P3s4', 'P1s4', 'P2s4'],
    ['P1s1', 'P3s3', 'P2s3', 'P1s3', 'P3s4', 'P4s4', 'P1s4', 'P2s4', 'P4s3', 'P2s1', 'P4s1', 'P3s1', 'P1s2', 'P2s2', 'P4s2', 'P3s2'],
    ['P2s2', 'P4s3', 'P2s3', 'P3s3', 'P4s4', 'P1s2', 'P4s2', 'P2s4', 'P1s4', 'P3s4', 'P1s3', 'P4s1', 'P3s1', 'P1s1', 'P3s2', 'P2s1'],
    ['P4s1', 'P1s1', 'P3s1', 'P1s2', 'P2s1', 'P3s2', 'P2s2', 'P4s3', 'P3s3', 'P4s4', 'P2s4', 'P1s4', 'P3s4', 'P1s3', 'P2s3', 'P4s2'],
    ['P4s4', 'P1s3', 'P3s3', 'P2s2', 'P1s2', 'P3s1', 'P2s1', 'P3s2', 'P1s1', 'P4s1', 'P2s3', 'P4s2', 'P1s4', 'P3s4', 'P2s4', 'P4s3'],
    ['P1s3', 'P4s4', 'P3s4', 'P2s4', 'P4s2', 'P2s2', 'P3s3', 'P1s1', 'P3s1', 'P1s2', 'P3s2', 'P4s3', 'P1s4', 'P2s3', 'P4s1', 'P2s1'],
    ['P3s2', 'P4s3', 'P2s1', 'P1s1', 'P3s1', 'P4s1', 'P1s3', 'P2s2', 'P1s2', 'P4s4', 'P2s4', 'P1s4', 'P3s4', 'P2s3', 'P3s3', 'P4s2'],
    ['P2s3', 'P4s2', 'P2s1', 'P4s4', 'P2s2', 'P1s2', 'P3s1', 'P1s1', 'P3s3', 'P4s3', 'P3s2', 'P4s1', 'P1s3', 'P3s4', 'P1s4', 'P2s4'],
    ['P2s2', 'P3s2', 'P4s1', 'P3s1', 'P2s1', 'P1s2', 'P4s4', 'P1s1', 'P4s3', 'P2s3', 'P3s3', 'P1s3', 'P3s4', 'P1s4', 'P4s2', 'P2s4'],
    ['P4s4', 'P2s2', 'P1s3', 'P3s3', 'P1s4', 'P3s4', 'P2s4', 'P4s2', 'P1s2', 'P4s1', 'P3s1', 'P1s1', 'P3s2', 'P2s1', 'P4s3', 'P2s3'],
    ['P2s2', 'P3s2', 'P1s1', 'P3s1', 'P2s1', 'P3s3', 'P4s1', 'P2s3', 'P1s4', 'P4s4', 'P2s4', 'P3s4', 'P1s3', 'P4s3', 'P1s2', 'P4s2'],
    ['P2s3', 'P3s3', 'P2s2', 'P1s3', 'P3s2', 'P1s2', 'P3s1', 'P4s1', 'P1s1', 'P2s1', 'P4s4', 'P1s4', 'P4s3', 'P3s4', 'P2s4', 'P4s2'],
    ['P2s4', 'P1s4', 'P3s3', 'P1s1', 'P3s1', 'P4s1', 'P2s2', 'P3s2', 'P4s3', 'P1s3', 'P3s4', 'P2s3', 'P4s4', 'P2s1', 'P4s2', 'P1s2'],
    ['P3s1', 'P1s1', 'P4s4', 'P2s1', 'P3s3', 'P4s1', 'P1s2', 'P4s2', 'P1s4', 'P2s3', 'P3s4', 'P1s3', 'P2s2', 'P3s2', 'P4s3', 'P2s4'],
    ['P2s2', 'P4s4', 'P2s1', 'P4s3', 'P2s4', 'P1s4', 'P4s2', 'P3s4', 'P2s3', 'P1s3', 'P3s3', 'P1s2', 'P3s2', 'P4s1', 'P3s1', 'P1s1'],
    ['P3s2', 'P1s3', 'P2s3', 'P4s2', 'P2s4', 'P1s4', 'P3s3', 'P1s1', 'P2s1', 'P4s4', 'P3s4', 'P4s3', 'P1s2', 'P3s1', 'P4s1', 'P2s2'],
    ['P1s4', 'P2s3', 'P4s4', 'P3s3', 'P1s1', 'P4s1', 'P3s1', 'P2s1', 'P4s3', 'P1s3', 'P3s4', 'P2s4', 'P4s2', 'P1s2', 'P3s2', 'P2s2'],
    ['P1s1', 'P3s1', 'P2s1', 'P3s3', 'P2s3', 'P4s2', 'P3s4', 'P1s4', 'P2s4', 'P4s4', 'P1s2', 'P2s2', 'P4s1', 'P3s2', 'P1s3', 'P4s3'],
    ['P1s4', 'P2s4', 'P4s2', 'P3s4', 'P2s3', 'P4s1', 'P3s1', 'P1s2', 'P2s1', 'P4s4', 'P3s3', 'P1s1', 'P4s3', 'P1s3', 'P3s2', 'P2s2'],
    ['P1s1', 'P3s1', 'P2s1', 'P3s2', 'P4s1', 'P2s3', 'P1s3', 'P3s3', 'P1s2', 'P4s2', 'P2s2', 'P4s4', 'P3s4', 'P1s4', 'P2s4', 'P4s3'],
    ['P1s3', 'P2s2', 'P3s2', 'P2s1', 'P4s3', 'P1s1', 'P4s1', 'P3s1', 'P1s2', 'P4s2', 'P1s4', 'P3s3', 'P4s4', 'P2s3', 'P3s4', 'P2s4'],
    ['P3s1', 'P1s2', 'P4s4', 'P1s4', 'P4s3', 'P2s2', 'P4s1', 'P2s1', 'P1s1', 'P3s3', 'P2s3', 'P1s3', 'P3s4', 'P2s4', 'P4s2', 'P3s2'],
    ['P4s4', 'P1s1', 'P3s1', 'P2s1', 'P3s2', 'P4s1', 'P1s2', 'P4s2', 'P3s4', 'P2s4', 'P1s4', 'P3s3', 'P2s2', 'P1s3', 'P4s3', 'P2s3'],
    ['P1s1', 'P4s1', 'P3s1', 'P2s1', 'P3s2', 'P4s2', 'P2s4', 'P4s4', 'P1s2', 'P2s2', 'P4s3', 'P2s3', 'P3s4', 'P1s3', 'P3s3', 'P1s4'],
    ['P2s4', 'P3s4', 'P4s3', 'P1s3', 'P2s2', 'P4s1', 'P3s2', 'P1s2', 'P2s1', 'P4s2', 'P1s4', 'P2s3', 'P4s4', 'P3s3', 'P1s1', 'P3s1'],
    ['P2s4', 'P4s3', 'P1s2', 'P3s2', 'P2s2', 'P3s3', 'P4s1', 'P3s1', 'P1s1', 'P2s1', 'P4s2', 'P2s3', 'P3s4', 'P1s3', 'P4s4', 'P1s4'],
    ['P2s2', 'P1s3', 'P4s1', 'P3s1', 'P2s1', 'P1s1', 'P3s2', 'P1s2', 'P3s3', 'P4s3', 'P3s4', 'P1s4', 'P4s4', 'P2s4', 'P4s2', 'P2s3'],
    ['P2s4', 'P4s4', 'P2s2', 'P4s2', 'P3s4', 'P1s3', 'P2s3', 'P1s4', 'P4s3', 'P3s3', 'P1s2', 'P3s2', 'P1s1', 'P3s1', 'P2s1', 'P4s1'],
    ['P3s2', 'P2s1', 'P3s3', 'P1s1', 'P4s4', 'P2s2', 'P4s3', 'P1s2', 'P3s1', 'P4s1', 'P2s3', 'P4s2', 'P1s3', 'P3s4', 'P2s4', 'P1s4'],
    ['P3s1', 'P4s1', 'P3s3', 'P2s2', 'P3s2', 'P1s1', 'P2s1', 'P1s2', 'P4s4', 'P3s4', 'P2s4', 'P4s3', 'P1s3', 'P2s3', 'P4s2', 'P1s4'],
    ['P2s3', 'P4s2', 'P2s4', 'P1s4', 'P4s4', 'P2s2', 'P4s3', 'P1s1', 'P3s2', 'P4s1', 'P3s1', 'P1s2', 'P2s1', 'P3s3', 'P1s3', 'P3s4'],
    ['P2s4', 'P4s2', 'P1s1', 'P3s1', 'P1s2', 'P3s2', 'P1s3', 'P3s4', 'P1s4', 'P4s4', 'P2s3', 'P3s3', 'P4s1', 'P2s2', 'P4s3', 'P2s1'],
    ['P2s1', 'P4s4', 'P1s3', 'P4s1', 'P1s2', 'P3s1', 'P1s1', 'P3s2', 'P2s2', 'P4s2', 'P3s3', 'P4s3', 'P1s4', 'P2s4', 'P3s4', 'P2s3'],
    ['P4s1', 'P3s3', 'P4s3', 'P2s4', 'P4s2', 'P1s3', 'P3s4', 'P2s3', 'P1s4', 'P4s4', 'P2s2', 'P1s2', 'P3s2', 'P1s1', 'P3s1', 'P2s1'],
    ['P4s3', 'P2s1', 'P1s1', 'P3s2', 'P2s2', 'P3s3', 'P1s4', 'P2s3', 'P3s4', 'P4s2', 'P2s4', 'P4s4', 'P1s3', 'P4s1', 'P3s1', 'P1s2'],
    ['P4s4', 'P1s2', 'P3s1', 'P2s1', 'P3s2', 'P2s2', 'P1s3', 'P3s4', 'P1s4', 'P4s3', 'P2s4', 'P4s2', 'P2s3', 'P4s1', 'P1s1', 'P3s3'],
    ['P1s1', 'P3s2', 'P1s2', 'P4s2', 'P2s2', 'P1s3', 'P4s3', 'P2s4', 'P1s4', 'P3s4', 'P4s4', 'P2s3', 'P3s3', 'P2s1', 'P3s1', 'P4s1'],
    ['P2s1', 'P3s1', 'P1s1', 'P3s2', 'P4s2', 'P2s4', 'P3s4', 'P4s4', 'P1s2', 'P2s2', 'P1s3', 'P4s1', 'P3s3', 'P2s3', 'P1s4', 'P4s3'],
    ['P2s4', 'P4s4', 'P1s2', 'P4s2', 'P3s3', 'P2s1', 'P3s2', 'P1s3', 'P2s3', 'P1s4', 'P3s4', 'P4s3', 'P2s2', 'P4s1', 'P3s1', 'P1s1'],
    ['P2s2', 'P3s3', 'P2s3', 'P1s4', 'P3s4', 'P4s2', 'P1s2', 'P2s1', 'P3s1', 'P4s1', 'P1s3', 'P3s2', 'P4s3', 'P2s4', 'P4s4', 'P1s1'],
    ['P4s3', 'P2s2', 'P3s3', 'P4s2', 'P2s4', 'P3s4', 'P1s4', 'P2s3', 'P1s3', 'P4s1', 'P2s1', 'P3s1', 'P1s1', 'P3s2', 'P1s2', 'P4s4'],
    ['P3s1', 'P4s1', 'P3s2', 'P1s1', 'P4s2', 'P2s4', 'P1s4', 'P2s3', 'P3s4', 'P4s4', 'P1s2', 'P2s2', 'P1s3', 'P4s3', 'P2s1', 'P3s3'],
    ['P2s4', 'P3s4', 'P1s4', 'P2s3', 'P4s3', 'P1s2', 'P3s2', 'P1s1', 'P2s1', 'P3s1', 'P4s1', 'P1s3', 'P2s2', 'P4s2', 'P3s3', 'P4s4'],
    ['P2s1', 'P4s2', 'P1s3', 'P3s3', 'P4s3', 'P1s2', 'P4s1', 'P2s3', 'P1s4', 'P3s4', 'P2s4', 'P4s4', 'P2s2', 'P3s2', 'P1s1', 'P3s1'],
    ['P3s3', 'P1s1', 'P3s1', 'P2s1', 'P4s4', 'P1s2', 'P4s3', 'P3s2', 'P4s2', 'P2s4', 'P1s4', 'P3s4', 'P1s3', 'P2s3', 'P4s1', 'P2s2'],
    ['P2s3', 'P3s4', 'P4s3', 'P2s1', 'P1s1', 'P3s1', 'P1s2', 'P3s3', 'P4s1', 'P2s2', 'P4s2', 'P3s2', 'P1s3', 'P4s4', 'P2s4', 'P1s4'],
    ['P1s4', 'P2s4', 'P4s2', 'P1s3', 'P3s4', 'P4s3', 'P3s2', 'P2s2', 'P1s2', 'P3s3', 'P2s3', 'P4s1', 'P3s1', 'P1s1', 'P2s1', 'P4s4'],
    ['P1s1', 'P3s3', 'P1s2', 'P2s1', 'P3s1', 'P4s1', 'P3s2', 'P4s3', 'P2s2', 'P1s3', 'P4s4', 'P3s4', 'P4s2', 'P2s4', 'P1s4', 'P2s3'],
    ['P2s2', 'P1s2', 'P3s1', 'P2s1', 'P1s1', 'P4s3', 'P3s2', 'P4s1', 'P2s3', 'P4s2', 'P3s3', 'P1s4', 'P2s4', 'P3s4', 'P1s3', 'P4s4'],
    ['P1s1', 'P3s2', 'P1s3', 'P4s4', 'P1s4', 'P4s3', 'P2s2', 'P4s2', 'P2s4', 'P3s4', 'P2s3', 'P3s3', 'P1s2', 'P2s1', 'P4s1', 'P3s1'],
    ['P1s3', 'P4s4', 'P2s2', 'P1s2', 'P3s2', 'P4s3', 'P2s4', 'P3s4', 'P1s4', 'P2s3', 'P3s3', 'P2s1', 'P4s2', 'P1s1', 'P4s1', 'P3s1'],
    ['P1s4', 'P2s4', 'P4s3', 'P3s4', 'P4s4', 'P2s2', 'P4s1', 'P1s3', 'P3s2', 'P1s1', 'P3s1', 'P1s2', 'P4s2', 'P2s1', 'P3s3', 'P2s3'],
    ['P2s3', 'P1s3', 'P4s2', 'P3s2', 'P4s1', 'P1s2', 'P4s3', 'P2s4', 'P1s4', 'P3s4', 'P4s4', 'P2s2', 'P3s3', 'P1s1', 'P3s1', 'P2s1'],
    ['P4s1', 'P3s1', 'P1s2', 'P4s4', 'P1s4', 'P2s4', 'P4s3', 'P1s1', 'P2s1', 'P3s3', 'P2s2', 'P4s2', 'P3s2', 'P1s3', 'P2s3', 'P3s4'],
    ['P1s4', 'P2s4', 'P3s4', 'P4s3', 'P2s2', 'P3s2', 'P2s1', 'P4s4', 'P1s2', 'P3s1', 'P1s1', 'P4s2', 'P1s3', 'P2s3', 'P3s3', 'P4s1'],
    ['P3s2', 'P1s1', 'P4s3', 'P1s3', 'P2s2', 'P1s2', 'P4s1', 'P3s1', 'P2s1', 'P4s4', 'P3s3', 'P4s2', 'P3s4', 'P2s3', 'P1s4', 'P2s4'],
    ['P4s3', 'P1s2', 'P4s1', 'P2s3', 'P3s4', 'P1s4', 'P4s4', 'P2s4', 'P4s2', 'P2s2', 'P3s3', 'P1s3', 'P3s2', 'P1s1', 'P3s1', 'P2s1'],
    ['P2s2', 'P4s1', 'P1s2', 'P3s3', 'P2s3', 'P1s3', 'P3s2', 'P4s3', 'P1s4', 'P4s2', 'P3s4', 'P2s4', 'P4s4', 'P2s1', 'P3s1', 'P1s1'],
    ['P2s2', 'P4s1', 'P3s1', 'P1s1', 'P4s3', 'P2s4', 'P3s4', 'P1s4', 'P4s4', 'P1s3', 'P4s2', 'P2s3', 'P3s3', 'P1s2', 'P2s1', 'P3s2'],
    ['P4s3', 'P1s4', 'P2s3', 'P3s4', 'P1s3', 'P2s2', 'P3s3', 'P4s1', 'P1s1', 'P3s2', 'P2s1', 'P3s1', 'P1s2', 'P4s2', 'P2s4', 'P4s4'],
    ['P3s1', 'P2s1', 'P1s1', 'P4s3', 'P2s2', 'P1s3', 'P4s1', 'P3s3', 'P4s2', 'P3s2', 'P1s2', 'P4s4', 'P2s4', 'P1s4', 'P2s3', 'P3s4'],
    ['P2s4', 'P1s4', 'P4s4', 'P1s3', 'P2s3', 'P3s4', 'P4s3', 'P1s1', 'P3s1', 'P4s1', 'P3s2', 'P1s2', 'P2s2', 'P3s3', 'P2s1', 'P4s2'],
    ['P4s2', 'P3s2', 'P2s1', 'P3s1', 'P1s2', 'P4s1', 'P1s3', 'P2s2', 'P4s4', 'P3s4', 'P2s4', 'P4s3', 'P1s4', 'P2s3', 'P3s3', 'P1s1'],
    ['P3s2', 'P2s2', 'P4s4', 'P3s3', 'P2s1', 'P4s1', 'P2s3', 'P4s2', 'P1s2', 'P3s1', 'P1s1', 'P4s3', 'P1s4', 'P2s4', 'P3s4', 'P1s3'],
    ['P2s2', 'P1s3', 'P4s1', 'P3s1', 'P2s1', 'P3s3', 'P4s2', 'P1s2', 'P3s2', 'P1s1', 'P4s4', 'P2s4', 'P1s4', 'P3s4', 'P2s3', 'P4s3'],
    ]
    
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
    
    # Create row and column indices
    rows = ['P{}'.format(i) for i in range(1, 5)]
    columns = ['s{}'.format(i) for i in range(1,5)]
    
    # Create DataFrame
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
        
        # Substract a value from the entire DataFrame
        selected_path_df = selected_path_df.sub(selected_path_df.iloc[0, 0])
        # Append
        selected_path_list.append(selected_path_df.values)
        
    # Calculate the mean of all the paths
    calibrated_times_sp = np.nanmean(selected_path_list, axis=0)
    calibration_times = calibrated_times_sp
    
    
    # Time calibration matrix calculated --------------------------------------
    print("---------------------------")
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
            mask = calibrated_data[f'{key}_T_sum_{j+1}'] != 0
            calibrated_data.loc[mask, f'{key}_T_sum_{j+1}'] += calibration_times[i][j]
    
    
    if create_plots:
        # Prepare a figure with 1x4 subplots (only for times, no positions)
        fig, axs = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    
        # Plot histograms for T_sum values from calibrated_data
        times = [calibrated_data[f'P1_T_sum_{i+1}'] for i in range(4)]
        titles_times = ["T_sum P1", "T_sum P2", "T_sum P3", "T_sum P4"]
    
        for i, (time, title) in enumerate(zip(times, titles_times)):
            time_non_zero = time[time != 0]  # Filter out zeros
            ax = axs[i]  # Access the subplot
            ax.hist(time_non_zero, bins=100, alpha=0.75)
            ax.set_title(title)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Frequency")
    
        # Show the figure for the T_sum histograms
        plt.suptitle('Histograms of T_sum Values for Each Plane', fontsize=16)
    
        if save_plots:
            name_of_file = 'Tsum_times_calibrated'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1

            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
    
        if show_plots: 
            plt.show()
        plt.close()
    
else:
    calibration_times = time_sum_reference
    calibrated_data['CRP_avg'] = 1000 # An extreme time to not crush the program
    print("Calibration in times was set to the reference! (calibration was not performed)\n", calibration_times)


print("----------------------------------------------------------------------")
print("----------------------- Time window filtering ------------------------")
print("----------------------------------------------------------------------")

# For each row, calculate the mean of the columns that have _T_sum_, then
# calculate the difference between each column and the mean, and finally
# check if the difference is within the time window of 7 ns

# Calculate the mean of the T_sum values for each row, considering only non-zero values
T_sum_columns = calibrated_data.filter(regex='_T_sum_')
mean_T_sum = T_sum_columns.apply(lambda row: row[row != 0].median() if row[row != 0].size > 0 else 0, axis=1)

# Calculate the difference between each T_sum value and the mean, but only for non-zero values
diff_T_sum = T_sum_columns.sub(mean_T_sum, axis=0)

# Check if the difference is within the time window, ignoring zero values
time_window_mask = np.abs(diff_T_sum) <= time_coincidence_window
time_window_mask[T_sum_columns == 0] = True  # Ignore zero values in the comparison

# Apply the mask to the data using .loc to avoid the SettingWithCopyWarning
calibrated_data.loc[:, T_sum_columns.columns] = T_sum_columns.where(time_window_mask, 0)

# Calculate how many values were set to zero
num_zeroed = (~time_window_mask).values.sum()
num_total = time_window_mask.size  # total number of elements

zeroed_percentage = num_zeroed / num_total

if zeroed_percentage > 0:
    print(f"Zeroed {zeroed_percentage:.2%} of the values outside the time window.")

global_variables['discarded_by_time_window_percentage'] = zeroed_percentage

if create_essential_plots or create_plots:

    t_sum_data = T_sum_columns.values  # shape: (n_events, n_detectors)
    widths = np.linspace(1, 40, 200)  # Scan range of window widths in ns

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
    counts_per_width_norm = counts_per_width / np.max(counts_per_width)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(widths, counts_per_width_norm, label='Normalized average count in window')
    # ax.fill_between(
    #     widths,
    #     (counts_per_width - counts_per_width_dev) / np.max(counts_per_width),
    #     (counts_per_width + counts_per_width_dev) / np.max(counts_per_width),
    #     alpha=0.2,
    #     label='±1σ band'
    # )
    ax.axvline(x=time_coincidence_window, color='red', linestyle='--', label='Time coincidence window')
    ax.set_xlabel("Window width (ns)")
    ax.set_ylabel("Normalized average # of T_sum values in window")
    ax.set_title("Fraction of hits within stat-centered window vs width")
    ax.grid(True)
    
    from scipy.optimize import curve_fit
    import numpy as np

    # Define model function: signal (logistic) + linear background
    def signal_plus_background(w, S, w0, tau, B):
        return S / (1 + np.exp(-(w - w0) / tau)) + B * w

    # Initial guess: [signal_height, center, width, background_slope]
    p0 = [1.0, 2.0, 0.5, 0.001]

    # Fit
    popt, pcov = curve_fit(signal_plus_background, widths, counts_per_width_norm, p0=p0)

    # Extract parameters
    S_fit, w0_fit, tau_fit, B_fit = popt
    print(f"Fit parameters:\n  Signal amplitude S = {S_fit:.4f}\n  Sigmoid center w0 = {w0_fit:.4f} ns\n  Sigmoid width τ = {tau_fit:.4f} ns\n  Background slope B = {B_fit:.6f} per ns")

    # Evaluate fit
    w_fit = np.linspace(min(widths), max(widths), 300)
    f_fit = signal_plus_background(w_fit, *popt)

    # Overlay fit curve
    ax.plot(w_fit, f_fit, 'k--', label='Signal + background fit')

    # Annotate signal and background
    ax.axhline(S_fit, color='green', linestyle=':', alpha=0.6, label=f'Signal plateau ≈ {S_fit:.2f}')

    # Compute stacked signal/background probabilities over window range
    s_vals = S_fit / (1 + np.exp(-(w_fit - w0_fit) / tau_fit))
    b_vals = B_fit * w_fit
    f_vals = s_vals + b_vals

    P_signal = s_vals / f_vals
    P_background = b_vals / f_vals

    # Create new axis above for stacked fill
    from matplotlib.gridspec import GridSpec

    # Reconstruct the figure with GridSpec
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.05)

    ax_fill = fig.add_subplot(gs[0])  # Top: signal vs. background fill
    ax_main = fig.add_subplot(gs[1], sharex=ax_fill)  # Bottom: your original plot

    # Fill signal/background areas
    ax_fill.fill_between(w_fit, 0, P_signal, color='green', alpha=0.4, label='Signal')
    ax_fill.fill_between(w_fit, P_signal, 1, color='red', alpha=0.4, label='Background')

    ax_fill.set_ylabel("Fraction")
    ax_fill.set_ylim(np.min(P_signal), 1)
    # ax_fill.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_fill.legend(loc="upper right")
    ax_fill.set_title("Estimated Signal and Background Fractions per Window Width")

    # Hide x-tick labels on top plot
    plt.setp(ax_fill.get_xticklabels(), visible=False)

    # Replot original data in ax_main
    ax_main.scatter(widths, counts_per_width_norm, label='Normalized average count in window')
    ax_main.axvline(x=time_coincidence_window, color='red', linestyle='--', label='Time coincidence window')
    ax_main.plot(w_fit, f_fit, 'k--', label='Signal + background fit')
    ax_main.axhline(S_fit, color='green', linestyle=':', alpha=0.6, label=f'Signal plateau ≈ {S_fit:.2f}')
    ax_main.set_xlabel("Window width (ns)")
    ax_main.set_ylabel("Normalized average # of T_sum values in window")
    ax_main.grid(True)
    fit_summary = (
        f"Fit: S = {S_fit:.3f}, w₀ = {w0_fit:.3f} ns, "
        f"τ = {tau_fit:.3f} ns, B = {B_fit:.4f}/ns"
    )
    ax_main.plot([], [], ' ', label=fit_summary)  # invisible handle to add text
    ax_main.legend()
    
    if save_plots:
        name_of_file = 'stat_window_accumulation'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    if show_plots:
        plt.show()
    plt.close()



print("----------------------------------------------------------------------")
print("---------------------- Y position calculation ------------------------")
print("----------------------------------------------------------------------")

if y_position_complex_method:
    print('Y position complex method.')
    # Initialize empty lists for y values
    y_values_M1 = []
    y_values_M2 = []
    y_values_M3 = []
    y_values_M4 = []

    # To store original y values before applying the lost bands or threshold adjustments
    original_y_values_M1 = []
    original_y_values_M2 = []
    original_y_values_M3 = []
    original_y_values_M4 = []

    def transformation(Q, exp):
        Q = np.where(Q <= 0, 0, Q)
        value = Q ** exp
        return value

    # Loop through each module to compute y values
    for module in ['P1', 'P2', 'P3', 'P4']:
        if module in ['P1', 'P3']:
            thick_strip = 4
            y_pos = y_pos_P1_and_P3
            y_width = y_width_P1_and_P3
            lost_band = [width - induction_section for width in y_width]  # Calculate lost band
        elif module in ['P2', 'P4']:
            thick_strip = 1
            y_pos = y_pos_P2_and_P4
            y_width = y_width_P2_and_P4
            lost_band = [width - induction_section for width in y_width]  # Calculate lost band
            
        lost_band = np.array(lost_band) / 2
        
        # Get the relevant Q_sum columns for the current module
        Q_sum_cols = [f'{module.replace("P", "Q")}_Q_sum_{i+1}' for i in range(4)]
        Q_sum_values = calibrated_data[Q_sum_cols].abs()

        Q_sum_trans = transformation(Q_sum_values, transf_exp)

        # Compute the sum of Q_sum values row-wise
        Q_sum_total = Q_sum_trans.sum(axis=1)

        # Calculate y using vectorized operations
        epsilon = 1e-10  # A small value to avoid division by very small numbers or zero
        y = (Q_sum_trans * y_pos).sum(axis=1) / (Q_sum_total + epsilon)

        # Save original y values for comparison later (without lost band adjustments)
        original_y_values = y.copy()

        # Check if y is too close to any of the y_pos values
        for i in range(len(y)):
            if Q_sum_total[i] == 0:
                continue  # Skip rows where Q_sum_total is 0
        
            # Check if the y value is too close to any y_pos value
            for j in range(len(y_pos)):
                # Check if within the threshold
                if abs(y[i] - y_pos[j]) < y_pos_threshold:
                    # Inside threshold: Generate a new value uniformly distributed in the lost band
                    lower_limit = y_pos[j] - lost_band[j]
                    upper_limit = y_pos[j] + lost_band[j]
                    
                    # Special case for strips in positions (1, 4) to extend uniformly to the strip border
                    if j == 0:  # Strip 1: Extend to the left edge of the detector
                        lower_limit = -np.sum(y_width) / 2
                    elif j == len(y_pos) - 1:  # Strip 4: Extend to the right edge of the detector
                        upper_limit = np.sum(y_width) / 2
                    
                    y[i] = np.random.uniform(lower_limit, upper_limit)
                    
                elif y_pos_threshold <= abs(y[i] - y_pos[j]) < y_width[j] / 2:
                    # Values between threshold and strip border are scaled between the lost band and the strip border
                    lower_limit = y_pos[j] - y_width[j] / 2
                    upper_limit = y_pos[j] + y_width[j] / 2
                    lost_band_value = lost_band[j]

                    if y[i] > y_pos[j]:
                        # Scale y[i] to fit between lost band and border (right side)
                        scaled_value = np.interp(
                            y[i],
                            [y_pos[j] + y_pos_threshold, upper_limit],
                            [y_pos[j] + lost_band_value, upper_limit]
                        )
                        y[i] = scaled_value
                    else:
                        # Scale y[i] to fit between lost band and border (left side)
                        scaled_value = np.interp(
                            y[i],
                            [lower_limit, y_pos[j] - y_pos_threshold],
                            [lower_limit, y_pos[j] - lost_band_value]
                        )
                        y[i] = scaled_value

        # Store the computed y values in the corresponding list
        if module == "P1":
            y_values_P1 = y
            original_y_values_P1 = original_y_values
        elif module == "P2":
            y_values_P2 = y
            original_y_values_P2 = original_y_values
        elif module == "P3":
            y_values_P3 = y
            original_y_values_P3 = original_y_values
        elif module == "P4":
            y_values_P4 = y
            original_y_values_P4 = original_y_values


# if uniform_y_method:
#     print('Y position uniform distribution method.')

#     # Initialize empty lists for y values
#     y_values_M1 = []
#     y_values_M2 = []
#     y_values_M3 = []
#     y_values_M4 = []

#     # Initialize original y-values for uniform method
#     original_y_values_M1 = []
#     original_y_values_M2 = []
#     original_y_values_M3 = []
#     original_y_values_M4 = []

#     # Loop through each module to compute y values
#     for module in ['P1', 'P2', 'P3', 'P4']:
#         if module in ['P1', 'P3']:
#             y_pos = y_pos_P1_and_P3
#             y_width = y_width_P1_and_P3
#         elif module in ['P2', 'P4']:
#             y_pos = y_pos_P2_and_P4
#             y_width = y_width_P2_and_P4

#         # Compute strip boundaries
#         strip_boundaries = [(center - width / 2, center + width / 2) for center, width in zip(y_pos, y_width)]
        
#         # Get the relevant Q_sum columns for the current module
#         Q_sum_cols = [f'{module.replace("T", "Q")}_Q_sum_{i+1}' for i in range(len(y_pos))]
#         Q_sum_values = calibrated_data[Q_sum_cols].abs()

#         # Compute the sum of Q_sum values row-wise
#         Q_sum_total = Q_sum_values.sum(axis=1)

#         # Initialize the y values for this module
#         y = np.zeros(len(calibrated_data))

#         # Loop through strips to generate uniform values
#         for j, (lower_limit, upper_limit) in enumerate(strip_boundaries):
#             # Generate uniform random values for the current strip
#             random_values = np.random.uniform(lower_limit, upper_limit, size=len(calibrated_data))

#             # Add random values only for rows where Q_sum for this strip is non-zero
#             y += random_values * (Q_sum_values.iloc[:, j] != 0)

#         # Store the computed y values in the corresponding list
#         if module == "P1":
#             y_values_M1 = y
#             original_y_values_M1 = y.copy()  # Store original values
#         elif module == "P2":
#             y_values_M2 = y
#             original_y_values_M2 = y.copy()  # Store original values
#         elif module == "P3":
#             y_values_M3 = y
#             original_y_values_M3 = y.copy()  # Store original values
#         elif module == "P4":
#             y_values_M4 = y
#             original_y_values_M4 = y.copy()  # Store original values


if uniform_y_method:
    print('Y position uniform distribution method.')

    # Initialize empty lists for y values
    y_values_M1, y_values_M2, y_values_M3, y_values_M4 = [], [], [], []
    original_y_values_M1, original_y_values_M2, original_y_values_M3, original_y_values_M4 = [], [], [], []

    # Loop through each module to compute y values
    for module in ['P1', 'P2', 'P3', 'P4']:
        if module in ['P1', 'P3']:
            y_pos = y_pos_P1_and_P3
            y_width = y_width_P1_and_P3
        elif module in ['P2', 'P4']:
            y_pos = y_pos_P2_and_P4
            y_width = y_width_P2_and_P4

        # Compute strip boundaries
        strip_boundaries = [(center - width / 2, center + width / 2) for center, width in zip(y_pos, y_width)]

        # Get the relevant Q_sum columns for the current module
        Q_sum_cols = [f'{module.replace("P", "Q")}_Q_sum_{i+1}' for i in range(len(y_pos))]
        Q_sum_values = calibrated_data[Q_sum_cols].abs()

        # Find the index of the maximum Q_sum for each row
        max_indices = Q_sum_values.idxmax(axis=1).apply(lambda col: int(col.split('_')[-1]) - 1)

        # Compute y values based on the maximum Q_sum index
        y = np.array([np.random.uniform(strip_boundaries[j][0], strip_boundaries[j][1]) for j in max_indices])

        # Store the computed y values in the corresponding list
        if module == "P1":
            y_values_M1 = y
            original_y_values_M1 = y.copy()  # Store original values
        elif module == "P2":
            y_values_M2 = y
            original_y_values_M2 = y.copy()  # Store original values
        elif module == "P3":
            y_values_M3 = y
            original_y_values_M3 = y.copy()  # Store original values
        elif module == "P4":
            y_values_M4 = y
            original_y_values_M4 = y.copy()  # Store original values



if not uniform_y_method and not y_position_complex_method:
    print('Y position center of the strip method.')
    # Initialize empty lists for y values
    y_values_M1 = []
    y_values_M2 = []
    y_values_M3 = []
    y_values_M4 = []
    
    # Loop through each module to compute y values
    for module in ['P1', 'P2', 'P3', 'P4']:
        if module in ['P1', 'P3']:
            thick_strip = 4
            y_pos = y_pos_P1_and_P3
        elif module in ['P2', 'P4']:
            thick_strip = 1
            y_pos = y_pos_P2_and_P4
    
        # Get the relevant Q_sum columns for the current module
        Q_sum_cols = [f'{module.replace("P", "Q")}_Q_sum_{i+1}' for i in range(4)]
        Q_sum_values = calibrated_data[Q_sum_cols].abs()
    
        # Compute the sum of Q_sum values row-wise
        Q_sum_total = Q_sum_values.sum(axis=1)
    
        # Calculate y using vectorized operations
        y = (Q_sum_values * y_pos).sum(axis=1) / Q_sum_total
        y[Q_sum_total == 0] = 0  # Set y to 0 where Q_sum_total is 0
    
        # Store the computed y values in the corresponding list
        if module == "P1":
            y_values_M1 = y.values
        elif module == "P2":
            y_values_M2 = y.values
        elif module == "P3":
            y_values_M3 = y.values
        elif module == "P4":
            y_values_M4 = y.values
            


y_values_dict = {
    'Y_1': y_values_M1,
    'Y_2': y_values_M2,
    'Y_3': y_values_M3,
    'Y_4': y_values_M4
}

y_values_df = pd.DataFrame(y_values_dict, index=calibrated_data.index)
calibrated_data = pd.concat([calibrated_data, y_values_df], axis=1)
calibrated_data = calibrated_data.copy()


# Plot the old and new Y's ------------------------------------------------------
if create_plots and y_position_complex_method:
    bin_number = 'auto'

    # Create a 3x4 grid for the plots
    fig, axs = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
    y_columns = ['Y_1', 'Y_2', 'Y_3', 'Y_4']
    titles = ['Y1', 'Y2', 'Y3', 'Y4']
    strip_borders_P1_and_P3 = np.cumsum(np.append(0, y_width_P1_and_P3)) - np.sum(y_width_P1_and_P3) / 2
    strip_borders_P2_and_P4 = np.cumsum(np.append(0, y_width_P2_and_P4)) - np.sum(y_width_P2_and_P4) / 2
    centers_dict = {'Y1': y_pos_P1_and_P3, 'Y3': y_pos_P1_and_P3, 'Y2': y_pos_P2_and_P4, 'Y4': y_pos_P2_and_P4}
    borders_dict = {'Y1': strip_borders_P1_and_P3, 'Y3': strip_borders_P1_and_P3, 'Y2': strip_borders_P2_and_P4, 'Y4': strip_borders_P2_and_P4}

    for i, (y_col, title) in enumerate(zip(y_columns, titles)):
        y_processed, y_original = calibrated_data[y_col].values, [original_y_values_M1, original_y_values_M2, original_y_values_M3, original_y_values_M4][i]
        y_non_zero_processed, y_non_zero_original = y_processed[y_processed != 0], y_original[y_original != 0]

        # Plot processed y-values
        axs[0, i].hist(y_non_zero_processed, bins=bin_number, alpha=0.5, color='blue', label='Processed')
        axs[0, i].set(title=f'{title} (Processed)', xlabel='Position (units)', ylabel='Frequency', xlim=(-150, 150), yscale='log')

        # Plot original y-values
        axs[1, i].hist(y_non_zero_original, bins=bin_number, alpha=0.5, color='green', label='Original')
        axs[1, i].set(title=f'{title} (Original)', xlabel='Position (units)', ylabel='Frequency', xlim=(-150, 150), yscale='log')

        # Plot both processed and original together in the third row
        axs[2, i].hist(y_non_zero_processed, bins=bin_number, alpha=0.4, color='blue', label='Processed')
        axs[2, i].hist(y_non_zero_original, bins=bin_number, alpha=0.4, color='green', label='Original')
        axs[2, i].set(title=f'{title} (Processed & Original)', xlabel='Position (units)', ylabel='Frequency', xlim=(-150, 150), yscale='log')

        # Add continuous lines for strip centers and borders in all rows
        for ax in [axs[0, i], axs[1, i], axs[2, i]]:
            for center in centers_dict[title]:
                ax.axvline(center, color='blue', linestyle='-', alpha=0.7)
            for border in borders_dict[title]:
                ax.axvline(border, color='red', linestyle='--', alpha=0.7)
            for band_border in [center + np.array([-lost_band[j], lost_band[j]]) for j, center in enumerate(centers_dict[title])]:
                ax.axvline(band_border[0], color='purple', linestyle=':', alpha=0.7)
                ax.axvline(band_border[1], color='purple', linestyle=':', alpha=0.7)
            for center in centers_dict[title]:
                ax.axvspan(center - y_pos_threshold, center + y_pos_threshold, color='yellow', alpha=0.2)

    plt.suptitle('Histograms of Y positions with Logarithmic Y-Axis', fontsize=16)
    
    if save_plots:
        name_of_file = 'y_positions_complex'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: plt.show()
    plt.close()


# Plot the old and new Y's ------------------------------------------------------
if create_plots and uniform_y_method:
    print("Plotting the uniform Y position method results")
    
    bin_number = 'auto'

    # Create a 3x4 grid for the plots
    fig, axs = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
    y_columns = ['Y_1', 'Y_2', 'Y_3', 'Y_4']
    titles = ['Y1', 'Y2', 'Y3', 'Y4']
    strip_borders_P1_and_P3 = np.cumsum(np.append(0, y_width_P1_and_P3)) - np.sum(y_width_P1_and_P3) / 2
    strip_borders_P2_and_P4 = np.cumsum(np.append(0, y_width_P2_and_P4)) - np.sum(y_width_P2_and_P4) / 2
    centers_dict = {'Y1': y_pos_P1_and_P3, 'Y3': y_pos_P1_and_P3, 'Y2': y_pos_P2_and_P4, 'Y4': y_pos_P2_and_P4}
    borders_dict = {'Y1': strip_borders_P1_and_P3, 'Y3': strip_borders_P1_and_P3, 'Y2': strip_borders_P2_and_P4, 'Y4': strip_borders_P2_and_P4}

    for i, (y_col, title) in enumerate(zip(y_columns, titles)):
        y_processed, y_original = calibrated_data[y_col].values, [original_y_values_M1, original_y_values_M2, original_y_values_M3, original_y_values_M4][i]
        y_non_zero_processed, y_non_zero_original = y_processed[y_processed != 0], y_original[y_original != 0]

        # Plot processed y-values
        axs[0, i].hist(y_non_zero_processed, bins=bin_number, alpha=0.5, color='blue', label='Processed')
        axs[0, i].set(title=f'{title} (Processed)', xlabel='Position (units)', ylabel='Frequency', xlim=(-150, 150), yscale='log')

        # Plot original y-values
        axs[1, i].hist(y_non_zero_original, bins=bin_number, alpha=0.5, color='green', label='Original')
        axs[1, i].set(title=f'{title} (Original)', xlabel='Position (units)', ylabel='Frequency', xlim=(-150, 150), yscale='log')

        # Plot both processed and original together in the third row
        axs[2, i].hist(y_non_zero_processed, bins=bin_number, alpha=0.4, color='blue', label='Processed')
        axs[2, i].hist(y_non_zero_original, bins=bin_number, alpha=0.4, color='green', label='Original')
        axs[2, i].set(title=f'{title} (Processed & Original)', xlabel='Position (units)', ylabel='Frequency', xlim=(-150, 150), yscale='log')

        # Add continuous lines for strip centers and borders in all rows
        for ax in [axs[0, i], axs[1, i], axs[2, i]]:
            for center in centers_dict[title]:
                ax.axvline(center, color='blue', linestyle='-', alpha=0.7)
            for border in borders_dict[title]:
                ax.axvline(border, color='red', linestyle='--', alpha=0.7)

    plt.suptitle('Histograms of Y positions with Logarithmic Y-Axis', fontsize=16)
    
    if save_plots:
        name_of_file = 'y_positions_uniform'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: plt.show()
    plt.close()


if create_plots and y_position_complex_method == False and uniform_y_method == False: 
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    y_columns = ['Y_1', 'Y_2', 'Y_3', 'Y_4']
    titles = ['Y1', 'Y2', 'Y3', 'Y4']
    
    # Loop through each Y column and plot in the corresponding subplot
    for i, (y_col, title) in enumerate(zip(y_columns, titles)):
        y = calibrated_data[y_col].values
        y_non_zero = y[y != 0]  # Filter out zeros
        
        # Plot histogram
        axs[i].hist(y_non_zero, bins=300, alpha=0.5, label=title)
        axs[i].set_title(title)
        axs[i].set_xlabel('Time (units)')
        axs[i].set_ylabel('Frequency')
        axs[i].set_yscale('log')  # Set y-axis to logarithmic scale
        axs[i].legend()
        
    plt.suptitle('Histograms of Y positions with Logarithmic Y-Axis', fontsize=16)
    
    if save_plots:
        name_of_file = 'y_positions_standard'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: plt.show()
    plt.close()

print("Y position calculated.")


print("----------------------------------------------------------------------")
print("----------------- Setting the variables of each RPC ------------------")
print("----------------------------------------------------------------------")

exp_final_rpc = 1
weighting_threshold = 0.2

new_columns = {}

# Function to compute weighted or maximum charge-based values
def compute_transformed_values(T_sums, T_diffs, Q_sums, weighted):
    # Get maximum charge and apply threshold
    Q_max = np.max(Q_sums, axis=1)
    limit_value = weighting_threshold * Q_max[:, np.newaxis]
    Q_sums[Q_sums < limit_value] = 0

    # Compute the transformed charges
    Q_transformed = Q_sums ** exp_final_rpc
    Q_sum_axis = Q_transformed.sum(axis=1) + 1e-10

    # Calculate weighted sums
    weighted_T_sum = (T_sums * Q_transformed).sum(axis=1) / Q_sum_axis
    weighted_T_diff = (T_diffs * Q_transformed).sum(axis=1) / Q_sum_axis

    # Find the index of the maximum charge for each event
    Q_max_index = np.argmax(Q_sums, axis=1)
    T_sum_max_charge = T_sums[np.arange(len(T_sums)), Q_max_index]
    T_diff_max_charge = T_diffs[np.arange(len(T_diffs)), Q_max_index]

    # Select values based on weighting mode
    if weighted:
        return weighted_T_sum, weighted_T_diff
    else:
        return T_sum_max_charge, T_diff_max_charge


for i_plane in range(1, 5):
    # Generate relevant column names for current plane
    T_sum_cols = [f'T{i_plane}_T_sum_{i+1}' for i in range(4)]
    T_diff_cols = [f'T{i_plane}_T_diff_{i+1}' for i in range(4)]
    Q_sum_cols = [f'Q{i_plane}_Q_sum_{i+1}' for i in range(4)]

    # Extract and preprocess data for calculations
    T_sums, T_diffs, Q_sums = (
        calibrated_data[cols].astype(float).fillna(0).values
        for cols in (T_sum_cols, T_diff_cols, Q_sum_cols)
    )

    # Make a copy of original Q_sums for final Q_sum calculation
    Q_sums_og = Q_sums.copy()

    # Compute final values
    T_sum_final, T_diff_final = compute_transformed_values(T_sums, T_diffs, Q_sums, weighted)
    
    # Store results in the new_columns dictionary
    new_columns[f'P{i_plane}_T_sum_final'] = T_sum_final
    new_columns[f'P{i_plane}_T_diff_final'] = T_diff_final
    new_columns[f'P{i_plane}_Q_sum_final'] = Q_sums_og.sum(axis=1)
    
    # Save the charge in each strip
    for strip in range(1, 5):
        new_columns[f'Q_P{i_plane}s{strip}'] = Q_sums_og[:, strip-1]


# Create a new DataFrame from computed columns and concatenate with original data
new_columns_df = pd.DataFrame(new_columns, index=calibrated_data.index)
calibrated_data = pd.concat([calibrated_data, new_columns_df], axis=1).copy()


if create_plots:
    
    # First plot: check the time calibration ------------------
    new_data = calibrated_data.copy()
    mask_all_non_zero = (new_data['P1_Q_sum_final'] != 0) & \
                        (new_data['P2_Q_sum_final'] != 0) & \
                        (new_data['P3_Q_sum_final'] != 0) & \
                        (new_data['P4_Q_sum_final'] != 0)
    
    # Filter new_data to keep only rows where all Q_sum_final values are non-zero
    new_data = new_data[mask_all_non_zero].copy()
    
    # Subtract P1_T_sum_final from P2_T_sum_final, P3_T_sum_final, and P4_T_sum_final in one step
    cols_to_adjust = [f'P{i}_T_sum_final' for i in range(1, 5)]
    new_data[cols_to_adjust] = new_data[cols_to_adjust].subtract(new_data['P1_T_sum_final'], axis=0)

    # Plotting
    if create_plots or create_essential_plots:
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.flatten()  # Flatten the axes array to easily iterate
        num_bins = 150
        
        # Iterate over i_plane from 1 to 4
        for i_plane in range(1, 5):
            # Define the column names for this plane
            t_sum_col = f'P{i_plane}_T_sum_final'
            t_diff_col = f'P{i_plane}_T_diff_final'
            q_sum_col = f'P{i_plane}_Q_sum_final'
            y_col = f'Y_{i_plane}'
            
            # Filter components in all vectors in which t_sum_col is bigger in abs to 10
            new_data = new_data[abs(new_data[t_sum_col]) < 10]
            
            # Filter out values that are NaN or 0 before plotting from the new_data DataFrame
            axes[(i_plane-1)*4].hist(new_data[t_sum_col].replace(0, np.nan).dropna(), bins=num_bins, alpha=0.7)
            axes[(i_plane-1)*4].set_title(t_sum_col)

            axes[(i_plane-1)*4 + 1].hist(new_data[t_diff_col].replace(0, np.nan).dropna(), bins=num_bins, alpha=0.7)
            axes[(i_plane-1)*4 + 1].set_title(t_diff_col)

            axes[(i_plane-1)*4 + 2].hist(new_data[q_sum_col].replace(0, np.nan).dropna(), bins=num_bins, alpha=0.7)
            axes[(i_plane-1)*4 + 2].set_title(q_sum_col)
            axes[(i_plane-1)*4 + 2].set_yscale('log')  # Log scale for better visualization

            axes[(i_plane-1)*4 + 3].hist(new_data[y_col].replace(0, np.nan).dropna(), bins=num_bins, alpha=0.7)
            axes[(i_plane-1)*4 + 3].set_title(y_col)
        
        plt.tight_layout()
        plt.suptitle('RPC variables, four planes, substracted time to debug', fontsize=16)
        
        if save_plots:
            name_of_file = 'rpc_variables_tcal_debug'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1

            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        
        if show_plots: 
            plt.show()
        plt.close()
    
    # Second plot: not substracting the t_sum from the other planes ----------------
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()  # Flatten the axes array to easily iterate
    num_bins = 150
    
    # Iterate over i_plane from 1 to 4
    for i_plane in range(1, 5):
        # Define the column names for this plane
        t_sum_col = f'P{i_plane}_T_sum_final'
        t_diff_col = f'P{i_plane}_T_diff_final'
        q_sum_col = f'P{i_plane}_Q_sum_final'
        y_col = f'Y_{i_plane}'
        
        # Filter out values that are NaN or 0 before plotting
        axes[(i_plane-1)*4].hist(calibrated_data[t_sum_col].replace(0, np.nan).dropna(), bins=num_bins, alpha=0.7)
        axes[(i_plane-1)*4].set_title(t_sum_col)

        axes[(i_plane-1)*4 + 1].hist(calibrated_data[t_diff_col].replace(0, np.nan).dropna(), bins=num_bins, alpha=0.7)
        axes[(i_plane-1)*4 + 1].set_title(t_diff_col)

        axes[(i_plane-1)*4 + 2].hist(calibrated_data[q_sum_col].replace(0, np.nan).dropna(), bins=num_bins, alpha=0.7)
        axes[(i_plane-1)*4 + 2].set_title(q_sum_col)
        axes[(i_plane-1)*4 + 2].set_yscale('log')

        axes[(i_plane-1)*4 + 3].hist(calibrated_data[y_col].replace(0, np.nan).dropna(), bins=num_bins, alpha=0.7)
        axes[(i_plane-1)*4 + 3].set_title(y_col)
    
    plt.tight_layout()
    plt.suptitle('RPC variables, unfiltered')
    
    if save_plots:
        name_of_file = 'rpc_variables_unfiltered'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: plt.show()
    plt.close()
    

print("--------------------- Filter 6: calibrated data ----------------------")
for col in calibrated_data.columns:
    if 'T_sum_final' in col:
        calibrated_data[col] = np.where((calibrated_data[col] < T_sum_RPC_left) | (calibrated_data[col] > T_sum_RPC_right), 0, calibrated_data[col])
    if 'T_diff_final' in col:
        calibrated_data[col] = np.where((calibrated_data[col] < T_diff_RPC_left) | (calibrated_data[col] > T_diff_RPC_right), 0, calibrated_data[col])
    if 'Q_sum_final' in col:
        calibrated_data[col] = np.where((calibrated_data[col] < Q_RPC_left) | (calibrated_data[col] > Q_RPC_right), 0, calibrated_data[col])
    if 'Y_' in col:
        calibrated_data[col] = np.where((calibrated_data[col] < Y_RPC_left) | (calibrated_data[col] > Y_RPC_right), 0, calibrated_data[col])


# ----------------------------------------------------------------------------------------------------------------
if stratos_save and station == 1:
    print("Saving X and Y for stratos.")
    
    stratos_df = calibrated_data.copy()
    
    # Select columns that start with "Y_" or match "T<number>_T_diff_final"
    filtered_columns = [col for col in stratos_df.columns if col.startswith("Y_") or "_T_diff_final" in col or 'datetime' in col]

    # Create a new DataFrame with the selected columns
    filtered_stratos_df = stratos_df[filtered_columns].copy()

    # Rename "T<number>_T_diff_final" to "X_<number>" and multiply by 200
    filtered_stratos_df.rename(columns=lambda col: f'X_{col.split("_")[0][1:]}' if "_T_diff_final" in col else col, inplace=True)
    filtered_stratos_df.loc[:, filtered_stratos_df.columns.str.startswith("X_")] *= 200

    # Display the first few rows of the modified DataFrame
    # print(filtered_stratos_df.head())

    # Define the save path
    save_stratos = os.path.join(stratos_list_events_directory, f'stratos_data_{save_filename_suffix}.csv')

    # Save DataFrame to CSV (correcting the method name)
    filtered_stratos_df.to_csv(save_stratos, index=False, float_format="%.1f")

    # print(f"Stratos data saved successfully to: {save_stratos}")
# ----------------------------------------------------------------------------------------------------------------

if create_plots:
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()  # Flatten the axes array to easily iterate
    num_bins = 150
    
    # Iterate over i_plane from 1 to 4
    for i_plane in range(1, 5):
        # Define the column names for this plane
        t_sum_col = f'P{i_plane}_T_sum_final'
        t_diff_col = f'P{i_plane}_T_diff_final'
        q_sum_col = f'P{i_plane}_Q_sum_final'
        y_col = f'Y_{i_plane}'
        
        # Filter out values that are NaN or 0 before plotting
        axes[(i_plane-1)*4].hist(calibrated_data[t_sum_col].replace(0, np.nan).dropna(), bins=num_bins, alpha=0.7)
        axes[(i_plane-1)*4].set_title(t_sum_col)

        axes[(i_plane-1)*4 + 1].hist(calibrated_data[t_diff_col].replace(0, np.nan).dropna(), bins=num_bins, alpha=0.7)
        axes[(i_plane-1)*4 + 1].set_title(t_diff_col)

        axes[(i_plane-1)*4 + 2].hist(calibrated_data[q_sum_col][calibrated_data[q_sum_col] > 0].dropna(), bins=num_bins, alpha=0.7)
        axes[(i_plane-1)*4 + 2].set_title(q_sum_col)
        axes[(i_plane-1)*4 + 2].set_yscale('log')

        axes[(i_plane-1)*4 + 3].hist(calibrated_data[y_col].replace(0, np.nan).dropna(), bins=num_bins, alpha=0.7)
        axes[(i_plane-1)*4 + 3].set_title(y_col)
    
    plt.tight_layout()
    plt.suptitle('RPC variables, filtered')
    
    if save_plots:
        name_of_file = 'rpc_variables_filtered'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: plt.show()
    plt.close()


# Same for hexbin
if create_plots or create_essential_plots:
    fig, axes = plt.subplots(4, 6, figsize=(24, 20))  # Adjusting for 6 combinations
    axes = axes.flatten()  # Flatten the axes array to easily iterate
    
    # Iterate over i_plane from 1 to 4
    for i_plane in range(1, 5):
        # Define the column names for this plane
        t_sum_col = f'P{i_plane}_T_sum_final'
        t_diff_col = f'P{i_plane}_T_diff_final'
        q_sum_col = f'P{i_plane}_Q_sum_final'
        y_col = f'Y_{i_plane}'
        
        # Filter out rows where any of the variables are NaN or 0 for all comparisons
        valid_rows = calibrated_data[[t_sum_col, t_diff_col, q_sum_col, y_col]].replace(0, np.nan).dropna()

        t_sum = valid_rows[t_sum_col]
        t_diff = valid_rows[t_diff_col]
        q_sum = valid_rows[q_sum_col]
        y = valid_rows[y_col]
        
        cond = q_sum < 50
        t_sum = t_sum[cond]
        t_diff = t_diff[cond]
        q_sum = q_sum[cond]
        y = y[cond]

        # Hexbin plot for all combinations (6 combinations for each i_plane)
        axes[(i_plane-1)*6].hexbin(t_sum, t_diff, gridsize=50, cmap='turbo')
        axes[(i_plane-1)*6].set_title(f'{t_sum_col} vs {t_diff_col}')
        
        axes[(i_plane-1)*6 + 1].hexbin(t_sum, q_sum, gridsize=50, cmap='turbo')
        axes[(i_plane-1)*6 + 1].set_title(f'{t_sum_col} vs {q_sum_col}')
        
        axes[(i_plane-1)*6 + 2].hexbin(t_sum, y, gridsize=50, cmap='turbo')
        axes[(i_plane-1)*6 + 2].set_title(f'{t_sum_col} vs {y_col}')
        
        axes[(i_plane-1)*6 + 3].hexbin(t_diff, q_sum, gridsize=50, cmap='turbo')
        axes[(i_plane-1)*6 + 3].set_title(f'{t_diff_col} vs {q_sum_col}')
        
        axes[(i_plane-1)*6 + 4].hexbin(t_diff, y, gridsize=50, cmap='turbo')
        axes[(i_plane-1)*6 + 4].set_title(f'{t_diff_col} vs {y_col}')
        
        axes[(i_plane-1)*6 + 5].hexbin(q_sum, y, gridsize=50, cmap='turbo')
        axes[(i_plane-1)*6 + 5].set_title(f'{q_sum_col} vs {y_col}')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle('Hexbin Plots for All Variable Combinations by Plane', fontsize=16)
    
    if save_plots:
        name_of_file = 'rpc_variables_hexbin_combinations'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: plt.show()
    plt.close()


if create_plots or create_essential_plots:
    fig, axes = plt.subplots(4, 6, figsize=(24, 20))  # Adjusting for 6 combinations
    axes = axes.flatten()  # Flatten the axes array to easily iterate
    
    # Iterate over i_plane from 1 to 4
    for i_plane in range(1, 5):
        # Define the column names for this plane
        t_sum_col = f'P{i_plane}_T_sum_final'
        t_diff_col = f'P{i_plane}_T_diff_final'
        q_sum_col = f'P{i_plane}_Q_sum_final'
        y_col = f'Y_{i_plane}'
        
        # Filter out rows where any of the variables are NaN or 0 for all comparisons
        valid_rows = calibrated_data[[t_sum_col, t_diff_col, q_sum_col, y_col]].replace(0, np.nan).dropna()

        t_sum = valid_rows[t_sum_col]
        t_diff = valid_rows[t_diff_col]
        q_sum = valid_rows[q_sum_col]
        y = valid_rows[y_col]
        
        cond = q_sum > 100
        t_sum = t_sum[cond]
        t_diff = t_diff[cond]
        q_sum = q_sum[cond]
        y = y[cond]

        # Hexbin plot for all combinations (6 combinations for each i_plane)
        axes[(i_plane-1)*6].hexbin(t_sum, t_diff, gridsize=50, cmap='turbo')
        axes[(i_plane-1)*6].set_title(f'{t_sum_col} vs {t_diff_col}')
        
        axes[(i_plane-1)*6 + 1].hexbin(t_sum, q_sum, gridsize=50, cmap='turbo')
        axes[(i_plane-1)*6 + 1].set_title(f'{t_sum_col} vs {q_sum_col}')
        
        axes[(i_plane-1)*6 + 2].hexbin(t_sum, y, gridsize=50, cmap='turbo')
        axes[(i_plane-1)*6 + 2].set_title(f'{t_sum_col} vs {y_col}')
        
        axes[(i_plane-1)*6 + 3].hexbin(t_diff, q_sum, gridsize=50, cmap='turbo')
        axes[(i_plane-1)*6 + 3].set_title(f'{t_diff_col} vs {q_sum_col}')
        
        axes[(i_plane-1)*6 + 4].hexbin(t_diff, y, gridsize=50, cmap='turbo')
        axes[(i_plane-1)*6 + 4].set_title(f'{t_diff_col} vs {y_col}')
        
        axes[(i_plane-1)*6 + 5].hexbin(q_sum, y, gridsize=50, cmap='turbo')
        axes[(i_plane-1)*6 + 5].set_title(f'{q_sum_col} vs {y_col}')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle('STREAMERS. Hexbin Plots for All Variable Combinations by Plane', fontsize=16)
    
    if save_plots:
        name_of_file = 'streamer_rpc_variables_hexbin_combinations'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: plt.show()
    plt.close()


# Plotting for articles and presentations
# if presentation_plots:
#     new_data = calibrated_data.copy()
#     mask_all_non_zero = (new_data['P1_Q_sum_final'] != 0) & \
#                         (new_data['P2_Q_sum_final'] != 0) & \
#                         (new_data['P3_Q_sum_final'] != 0) & \
#                         (new_data['P4_Q_sum_final'] != 0)

#     # Filter new_data to keep only rows where all Q_sum_final values are non-zero
#     new_data = new_data[mask_all_non_zero].copy()

#     for plane in range(1, 5):  # Loop through all four planes
#         fig, ax = plt.subplots(1, 1, figsize=(5, 5))  # Small figsize for article column

#         # Define the column names for the specified plane
#         t_diff_col = f'P{plane}_T_diff_final'
#         y_col = f'Y_{plane}'

#         # Filter out rows where any of the variables are NaN or 0 for all comparisons
#         valid_rows = new_data[[t_diff_col, y_col]].replace(0, np.nan).dropna()

#         # Transform t_diff_col to X by multiplying with tdiff_to_x
#         valid_rows['X_transformed'] = valid_rows[t_diff_col] * tdiff_to_x

#         # Further filter data with abs value less than 150 for X_transformed
#         valid_rows = valid_rows[(abs(valid_rows['X_transformed']) < 150) & (abs(valid_rows[y_col]) < 150)]

#         # Hexbin plot for X_transformed vs Y
#         ax.hexbin(valid_rows['X_transformed'], valid_rows[y_col], gridsize=35, cmap='turbo')
#         cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', shrink=0.75)
#         cbar.set_label('Counts')
        
#         ax.set_xlabel('X / mm')
#         ax.set_ylabel('Y / mm')
#         plt.tight_layout()
#         ax.set_aspect('equal', adjustable='box')

#         if save_plots:
#             name_of_file = f'rpc_plane{plane}_x_y_hexbin'
#             final_filename = f'{fig_idx}_{name_of_file}.png'
#             fig_idx += 1
            
#             save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
#             plot_list.append(save_fig_path)
#             plt.savefig(save_fig_path, format='png', dpi=300)

#         if show_plots:
#             plt.show()

#         plt.close()

print("----------------------------------------------------------------------")
print("----------------------- Alternative fitting --------------------------")
print("----------------------------------------------------------------------")

# Function to fit a straight line in 3D
def fit_3d_line(X, Y, Z, sX, sY, sZ):
    """
    Least squares fitting of a 3D straight line.
    Returns theta (zenith), phi (azimuth), and chi-squared.
    """
    # Stack coordinates into an array
    points = np.vstack((X, Y, Z)).T
    
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Center data
    centered_points = points - centroid

    # Compute SVD (Singular Value Decomposition)
    _, _, Vt = np.linalg.svd(centered_points)
    
    # Direction vector (first principal component)
    direction_vector = Vt[0]

    # Extract direction components
    d_x, d_y, d_z = direction_vector

    # Compute theta (zenith) and phi (azimuth)
    theta = np.arccos(d_z / np.linalg.norm(direction_vector))
    phi = np.arctan2(d_y, d_x)
    
    if d_z != 0:  # Ensure no division by zero
        t_0 = -centroid[2] / d_z
        x_z0 = centroid[0] + t_0 * d_x
        y_z0 = centroid[1] + t_0 * d_y
    else:
        # x_z0, y_z0 = 0, 0 # Line is parallel to Z-plane
        x_z0, y_z0 = np.nan, np.nan # Line is parallel to Z-plane
    
    # # To degrees
    # theta = np.degrees(theta)
    # phi = np.degrees(phi)
    
    # Compute chi-squared
    distances = np.linalg.norm(np.cross(centered_points, direction_vector), axis=1)
    chi2 = np.sum((distances / np.sqrt(np.array(sX)**2 + np.array(sY)**2 + np.array(sZ)**2)) ** 2)

    return x_z0, y_z0, theta, phi, chi2

# Initialize results
alt_fit_results = ['alt_theta', 'alt_phi', 'alt_chi2']
alt_new_columns_df = pd.DataFrame(0., index=calibrated_data.index, columns=alt_fit_results)
calibrated_data = pd.concat([calibrated_data, alt_new_columns_df], axis=1)

# Loop over all tracks
for idx, track in calibrated_data.iterrows():
    planes_to_iterate = []
    
    # Identify valid planes with charge
    for i_plane in range(nplan):
        charge_plane = getattr(track, f'P{i_plane + 1}_Q_sum_final')
        if charge_plane > 4:
            planes_to_iterate.append(i_plane + 1)

    planes_to_iterate = np.array(planes_to_iterate)

    if len(planes_to_iterate) >= 2:  # Only fit if 2 or more points exist
        X, Y, Z, sX, sY, sZ = [], [], [], [], [], []

        for iplane in planes_to_iterate:
            t_d = getattr(track, f'P{iplane}_T_diff_final')
            x_p = strip_speed * t_d
            X.append(x_p)
            Y.append(getattr(track, f'Y_{iplane}'))
            Z.append(z_positions[iplane - 1])
            sX.append(anc_sx)
            sY.append(anc_sy)
            sZ.append(anc_sz)

        # Fit line
        x, y, theta, phi, chi2 = fit_3d_line(X, Y, Z, sX, sY, sZ)

        # Store results
        calibrated_data.at[idx, 'alt_x'] = x
        calibrated_data.at[idx, 'alt_y'] = y
        calibrated_data.at[idx, 'alt_theta'] = theta
        calibrated_data.at[idx, 'alt_phi'] = phi
        calibrated_data.at[idx, 'alt_chi2'] = chi2


if create_plots:
    # Scatter plot of alt_theta vs alt_phi
    # plt.figure(figsize=(8, 6))
    # plt.scatter(calibrated_data['alt_phi'], calibrated_data['alt_theta'], alpha=0.7, s = 1)
    # plt.xlabel('Azimuth (φ) [radians]')
    # plt.ylabel('Zenith (θ) [radians]')
    # plt.title('Scatter Plot of Fitted Angles')
    # plt.grid(True)
    # plt.show()
    
    # if save_plots:
    #     name_of_file = 'alternative_fitting_results_scatter_combination_projections'
    #     final_filename = f'{fig_idx}_{name_of_file}.png'
    #     fig_idx += 1

    #     save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
    #     plot_list.append(save_fig_path)
    #     plt.savefig(save_fig_path, format='png')

    # # Show plot if enabled
    # if show_plots:
    #     plt.show()

    # plt.close()
    
    # Contour plot of alt_chi2 vs alt_theta and alt_phi
    theta_values = calibrated_data['alt_theta'].values
    phi_values = calibrated_data['alt_phi'].values
    chi2_values = np.clip(calibrated_data['alt_chi2'].values, 0, 1)
    
    # Define grid resolution
    theta_bins = np.linspace(min(theta_values), max(theta_values), 40)
    phi_bins = np.linspace(min(phi_values), max(phi_values), 40)

    # Create a 2D histogram
    H, theta_edges, phi_edges = np.histogram2d(theta_values, phi_values, bins=[theta_bins, phi_bins], weights=chi2_values)
    counts, _, _ = np.histogram2d(theta_values, phi_values, bins=[theta_bins, phi_bins])

    # Compute the average chi2 in each bin
    with np.errstate(divide='ignore', invalid='ignore'):
        Chi2_binned = np.where(counts > 0, H / counts, 0)  # Average chi2 in each bin, 0 where no data

    # Define grid for plotting
    Theta_mid = (theta_edges[:-1] + theta_edges[1:]) / 2
    Phi_mid = (phi_edges[:-1] + phi_edges[1:]) / 2
    Theta_grid, Phi_grid = np.meshgrid(Theta_mid, Phi_mid, indexing='ij')

    # Plot binned contour
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Phi_grid, Theta_grid, Chi2_binned, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Chi-Squared')
    plt.xlabel('Azimuth (φ) [radians]')
    plt.ylabel('Zenith (θ) [radians]')
    plt.title('Binned Contour Plot of Chi-Squared')
    plt.grid(True)
    plt.show()
    
    print("Alternative fitting done and saving...")

    if save_plots:
        name_of_file = 'alternative_fitting_results_hexbin_combination_projections'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    # Show plot if enabled
    if show_plots:
        plt.show()

    plt.close()
    
    # Contour plot of alt_chi2 vs alt_theta and alt_phi
    # Filter Theta between 0 and 1.3
    plot_data = calibrated_data[(calibrated_data['alt_theta'] > 0) & (calibrated_data['alt_theta'] < 1.3)]
    theta_values = plot_data['alt_theta'].values
    phi_values = plot_data['alt_phi'].values
    chi2_values = np.clip(plot_data['alt_chi2'].values, 0, 1)
    
    # Define grid resolution
    theta_bins = np.linspace(min(theta_values), max(theta_values), 20)
    phi_bins = np.linspace(min(phi_values), max(phi_values), 20)

    # Create a 2D histogram
    H, theta_edges, phi_edges = np.histogram2d(theta_values, phi_values, bins=[theta_bins, phi_bins], weights=chi2_values)
    counts, _, _ = np.histogram2d(theta_values, phi_values, bins=[theta_bins, phi_bins])

    # Compute the average chi2 in each bin
    with np.errstate(divide='ignore', invalid='ignore'):
        Chi2_binned = np.where(counts > 0, H / counts, 0)  # Average chi2 in each bin, 0 where no data

    # Define grid for plotting
    Theta_mid = (theta_edges[:-1] + theta_edges[1:]) / 2
    Phi_mid = (phi_edges[:-1] + phi_edges[1:]) / 2
    Theta_grid, Phi_grid = np.meshgrid(Theta_mid, Phi_mid, indexing='ij')

    # Plot binned contour
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Phi_grid, Theta_grid, Chi2_binned, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Chi-Squared')
    plt.xlabel('Azimuth (φ) [radians]')
    plt.ylabel('Zenith (θ) [radians]')
    plt.title('Binned Contour Plot of Chi-Squared')
    plt.grid(True)
    plt.show()
    
    print("Alternative angle fitting done and saving...")

    if save_plots:
        name_of_file = 'alternative_fitting_2_results_hexbin_combination_projections'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    # Show plot if enabled
    if show_plots:
        plt.show()

    plt.close()


print("----------------------------------------------------------------------")
print("----------------- Alternative slowness fitting -----------------------")
print("----------------------------------------------------------------------")


# Initialize results
alt_fit_slow_results = ['alt_slowness']
alt_slow_new_columns_df = pd.DataFrame(0., index=calibrated_data.index, columns=alt_fit_slow_results)
calibrated_data = pd.concat([calibrated_data, alt_slow_new_columns_df], axis=1)

# Loop over all tracks
for idx, track in calibrated_data.iterrows():
    planes_to_iterate = []
    
    # Identify valid planes with charge
    for i_plane in range(nplan):
        charge_plane = getattr(track, f'P{i_plane + 1}_Q_sum_final')
        if charge_plane > 4:
            planes_to_iterate.append(i_plane + 1)

    planes_to_iterate = np.array(planes_to_iterate)
    
    if len(planes_to_iterate) >= 2:  # Only fit if 2 or more points exist
        tsum = []
        z = []
        
        for iplane in planes_to_iterate:
            t_s = getattr(track, f'P{iplane}_T_sum_final')
            tsum.append(t_s)
            z.append(z_positions[iplane - 1])

        theta = getattr(track, f'alt_theta')
        phi = getattr(track, f'alt_phi')
        
        # Now calculate the slowness using the difference of the time sums and the angle subtended by the trace
        
        # Convert to arrays
        tsum = np.array(tsum)
        z = np.array(z)

        # Projected track path length between planes along the track direction
        # dz = z - z[0]
        # dz_proj = dz / np.cos(theta)
        # slope, intercept = np.polyfit(dz_proj, tsum - tsum[0], deg=1)

        # Step 1: Build track direction vector
        v_dir = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        # Step 2: Build hit positions in 3D (only z is known, assume y=0, x=0)
        # You can adapt this if you know x/y per plane
        positions = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)

        # Step 3: Project each position onto the direction vector to get path length
        proj_dist = positions @ v_dir  # scalar projection: s = r · v̂

        # Step 4: Fit T_sum vs. projected distance
        slope, intercept = np.polyfit(proj_dist - proj_dist[0], tsum - tsum[0], deg=1)
        calibrated_data.at[idx, 'alt_slowness'] = slope
    
print("Alternative slowness fitting done and saving...")
    
if create_essential_plots or create_plots:
    # Histogram the slowness calculated
    plt.figure(figsize=(8, 6))
    v = calibrated_data['alt_slowness'].replace(0, np.nan).dropna()
    cond = (v > slowness_filter_left) & (v < slowness_filter_right)
    v = v[cond]
    plt.hist(v, bins=200, alpha=0.7)
    plt.xlabel('Slowness (ns/mm)')
    plt.ylabel('Counts')
    plt.title('Histogram of Slowness')
    plt.grid(True)
    plt.xlim(slowness_filter_left, slowness_filter_right)  # Adjust x-axis limits as needed
    # plt.ylim(slowness_filter_left, slowness_filter_right)  # Adjust y-axis limits as needed
    plt.tight_layout()
    
    if save_plots:
        name_of_file = 'alt_slowness'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    # Show plot if enabled
    if show_plots:
        plt.show()
    plt.close()

# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

print("----------------------------------------------------------------------")
print("------------------------- TimTrack fitting ---------------------------")
print("----------------------------------------------------------------------")

if fixed_speed:
    print("Fixed the slowness to the speed of light.")
else:
    print("Slowness not fixed.")

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

def fs2(nvar, npar, vs, vdat, vsig, lenx, ss, zi):
    va = np.zeros(npar)
    mk = fmkx(nvar, npar, vs, vsig, ss, zi)
    va = fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
    vm = fvmx(nvar, vs, lenx, ss, zi)
    mg = fmgx(nvar, npar, vs, ss, zi)
    mw = fmwx(nvar, vsig)
    vg = vm - mg @ vs
    vdmg = vdat - vg
    mg = fmgx(nvar, npar, vs, ss, zi)
    sk = vs.transpose() @ mk @ vs
    sa = vs.transpose() @ va
    s0 = vdmg.transpose() @ mw @ vdmg
    s2 = sk - 2*sa + s0
    return s2

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
    tffit = T0 + S0 * kz * zi + (lenx/2 + xfit) * ss
    tbfit = T0 + S0 * kz * zi + (lenx/2 - xfit) * ss
    tsfit = 0.5 * (tffit + tbfit)
    tdfit = 0.5 * (tffit - tbfit)
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

if fixed_speed:
    npar = 5
else:
    npar = 6
nvar = 3

i = 0
ntrk  = len(calibrated_data)
if limit and limit_number < ntrk: ntrk = limit_number
print("-----------------------------")
print(f"{ntrk} events to be fitted")

if res_ana_removing_planes:
    timtrack_results = ['x', 'xp', 'y', 'yp', 't0', 's',
                    'th_chi', 'res_y', 'res_ts', 'res_td', 'type',
                    'res_ystr_1', 'res_ystr_2', 'res_ystr_3', 'res_ystr_4',
                    'res_tsum_1', 'res_tsum_2', 'res_tsum_3', 'res_tsum_4',
                    'res_tdif_1', 'res_tdif_2', 'res_tdif_3', 'res_tdif_4',
                    'ext_res_ystr_1', 'ext_res_ystr_2', 'ext_res_ystr_3', 'ext_res_ystr_4',
                    'ext_res_tsum_1', 'ext_res_tsum_2', 'ext_res_tsum_3', 'ext_res_tsum_4',
                    'ext_res_tdif_1', 'ext_res_tdif_2', 'ext_res_tdif_3', 'ext_res_tdif_4']
else:
    timtrack_results = ['x', 'xp', 'y', 'yp', 't0', 's', 'type', 'th_chi',
                    'charge_1', 'charge_2', 'charge_3', 'charge_4', 'charge_event',
                    'res_ystr_1', 'res_ystr_2', 'res_ystr_3', 'res_ystr_4',
                    'res_tsum_1', 'res_tsum_2', 'res_tsum_3', 'res_tsum_4',
                    'res_tdif_1', 'res_tdif_2', 'res_tdif_3', 'res_tdif_4']

new_columns_df = pd.DataFrame(0., index=calibrated_data.index, columns=timtrack_results)
calibrated_data = pd.concat([calibrated_data, new_columns_df], axis=1)

# TimTrack starts ------------------------------------------------------
repeat = number_of_TT_executions - 1 if timtrack_iteration else 0
for iteration in range(repeat + 1):
    calibrated_data.loc[:, timtrack_results] = 0.0
    
    fitted = 0
    print("-----------------------------")
    print(f"TimTrack iteration {iteration}")
    print("-----------------------------")
    
    if crontab_execution:
        iterator = calibrated_data.iterrows()
    else:
        iterator = tqdm(calibrated_data.iterrows(), total=calibrated_data.shape[0], desc="Processing events")
    
    for idx, track in iterator:
        
        # if idx == 1:
        #     print("First event saved to file.")
        #     # Save to csv
        #     track.to_csv('track.csv')
        
        # INTRODUCTION ------------------------------------------------------------------
        track_numeric = pd.to_numeric(track.drop('datetime'), errors='coerce')
        
        # -------------------------------------------------------------------------------
        name_type = ""
        planes_to_iterate = []
        
        charge_event = 0
        for i_plane in range(nplan):
            # Check if the sum of the charges in the current plane is non-zero
            charge_plane = getattr(track, f'P{i_plane + 1}_Q_sum_final')
            if charge_plane != 0:
                # Append the plane number to name_type and planes_to_iterate
                name_type += f'{i_plane + 1}'
                planes_to_iterate.append(i_plane + 1)
                calibrated_data.at[idx, f'charge_{i_plane + 1}'] = charge_plane
                charge_event += charge_plane
                
        calibrated_data.at[idx, 'charge_event'] = charge_event
        planes_to_iterate = np.array(planes_to_iterate)
        
        # FITTING -----------------------------------------------------------------------
        if len(planes_to_iterate) > 1:
            if fixed_speed:
                vs  = np.asarray([0,0,0,0,0])
            else:
                vs  = np.asarray([0,0,0,0,0,sc])
            mk  = np.zeros([npar, npar])
            va  = np.zeros(npar)
            istp = 0   # nb. of fitting steps
            dist = d0
            while dist>cocut:
                # for iplane, istrip in zip(planes_to_iterate, istrip_list):
                for iplane in planes_to_iterate:
                    
                    # Data --------------------------------------------------------
                    zi  = z_positions[iplane - 1]                              # z pos
                    yst = getattr(track, f'Y_{iplane}')                        # y position
                    sy  = anc_sy                                               # uncertainty in y               
                    ts  = getattr(track, f'P{iplane}_T_sum_final')             # t sum
                    sts = anc_std                                              # uncertainty in t sum
                    td  = getattr(track, f'P{iplane}_T_diff_final')            # t dif
                    std = anc_std                                              # uncertainty in tdif
                    # -------------------------------------------------------------
                    
                    vdat = [yst, ts, td]
                    vsig = [sy, sts, std]
                    mk = mk + fmkx(nvar, npar, vs, vsig, ss, zi)
                    va = va + fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
                istp = istp + 1
                merr = linalg.inv(mk)     # Error matrix
                vs0 = vs
                vs  = merr @ va          # sEa equation
                dist = fmahd(npar, vs, vs0, merr)
                if istp > 5:
                    continue
            dist = 10
            vsf = vs       # final saeta
            fitted += 1
        else:
            continue
        
        
        # RESIDUAL ANALYSIS ----------------------------------------------------------------------------
        
        # Standard residual analysis
        # chi2 = fs2(nvar, npar, vsf, vdat, vsig, lenx, ss, zi) # Theoretical chisq
        # chi2 = 0 # Theoretical chisq
        
        # Fit residuals
        res_ystr = 0
        res_tsum = 0
        res_tdif = 0
        ndat     = 0
        
        if len(planes_to_iterate) > 1:
            # for iplane, istrip in zip(planes_to_iterate, istrip_list):
            for iplane in planes_to_iterate:
                
                ndat = ndat + nvar
                
                # Data --------------------------------------------------------
                zi  = z_positions[iplane - 1]                                  # z pos
                yst = getattr(track, f'Y_{iplane}')                            # y position
                sy  = anc_sy                                                   # uncertainty in y               
                ts  = getattr(track, f'P{iplane}_T_sum_final')                 # t sum
                sts = anc_std                                                  # uncertainty in t sum
                td  = getattr(track, f'P{iplane}_T_diff_final')                # t dif
                std = anc_std                                                  # uncertainty in tdif
                # -------------------------------------------------------------
                
                vdat = [yst, ts, td]
                vsig = [sy, sts, std]
                vres = fres(vsf, vdat, lenx, ss, zi)
                
                calibrated_data.at[idx, f'res_ystr_{iplane}'] = vres[0]
                calibrated_data.at[idx, f'res_tsum_{iplane}'] = vres[1]
                calibrated_data.at[idx, f'res_tdif_{iplane}'] = vres[2]
                
                res_ystr  = res_ystr  + vres[0]
                res_tsum  = res_tsum + vres[1]
                res_tdif  = res_tdif + vres[2]
                
            ndf  = ndat - npar    # number of degrees of freedom; was ndat - npar
            
            # calibrated_data.at[idx, f'res_ystr_{iplane}'] = res_ystr
            # calibrated_data.at[idx, f'res_tsum_{iplane}'] = res_tsum
            # calibrated_data.at[idx, f'res_tdif_{iplane}'] = res_tdif
            
            calibrated_data.at[idx, 'type'] = builtins.int(name_type)
            
            chi2 = res_ystr**2 + res_tsum**2 + res_tdif**2
            calibrated_data.at[idx, 'th_chi'] = chi2
            
            calibrated_data.at[idx, 'x'] = vsf[0]
            calibrated_data.at[idx, 'xp'] = vsf[1]
            calibrated_data.at[idx, 'y'] = vsf[2]
            calibrated_data.at[idx, 'yp'] = vsf[3]
            calibrated_data.at[idx, 't0'] = vsf[4]
            
            if fixed_speed:
                calibrated_data.at[idx, 's'] = sc
            else:
                calibrated_data.at[idx, 's'] = vsf[5]
        
        # Residual analysis with 4-plane tracks (hide a plane and make a fit in the 3 remaining planes)
        if len(planes_to_iterate) == 4 and res_ana_removing_planes:
            
            # for iplane_ref, istrip_ref in zip(planes_to_iterate, istrip_list):
            for iplane_ref in planes_to_iterate:
                
                # Data --------------------------------------------------------
                z_ref  = z_positions[iplane_ref - 1]                               # z pos
                y_strip_ref = getattr(track, f'Y_{iplane_ref}')                    # y position
                sy  = anc_sy                                                       # uncertainty in y
                t_sum_ref  = getattr(track, f'P{iplane_ref}_T_sum_final')          # t sum
                sts = anc_sts                                                      # uncertainty in t sum
                t_dif_ref  = getattr(track, f'P{iplane_ref}_T_diff_final')         # t dif
                std = anc_std                                                      # uncertainty in tdif
                # -----------------------------------------------------------------
                
                vdat_ref = [ y_strip_ref, t_sum_ref, t_dif_ref]
                
                # istrip_list_short = istrip_list[ planes_to_iterate != iplane_ref ]
                planes_to_iterate_short = planes_to_iterate[planes_to_iterate != iplane_ref]
                
                vs     = vsf  # We start with the previous 4-planes fit
                mk     = np.zeros([npar, npar])
                va     = np.zeros(npar)
                isP3 = 0
                dist = d0
                while dist>cocut:
                    # for iplane, istrip in zip(planes_to_iterate_short, istrip_list_short):
                    for iplane in planes_to_iterate_short:
                    
                        # Data --------------------------------------------------------
                        zi  = z_positions[iplane - 1] - z_ref                           # z pos
                        yst = getattr(track, f'Y_{iplane}')                             # y position
                        sy  = anc_sy                                                    # uncertainty in y
                        ts  = getattr(track, f'P{iplane}_T_sum_final')                  # t sum
                        sts = anc_sts                                                   # uncertainty in t sum
                        td  = getattr(track, f'P{iplane}_T_diff_final')                 # t dif
                        std = anc_std                                                   # uncertainty in tdif
                        # -------------------------------------------------------------
                        
                        vdat = [yst, ts, td]
                        vsig = [sy, sts, std]
                        mk = mk + fmkx(nvar, npar, vs, vsig, ss, zi)
                        va = va + fvax(nvar, npar, vs, vdat, vsig, lenx, ss, zi)
                    isP3 = isP3 + 1
                    merr = linalg.inv(mk)    # Error matrix
                    vs0 = vs
                    vs  = merr @ va          # sEa equation
                    dist = fmahd(npar, vs, vs0, merr)
                    
                vsig = [sy, sts, std]
                # v_track  = [ iplane_ref, istrip_ref ]
                v_res    = fres(vs, vdat_ref, lenx, ss, 0)
                
                calibrated_data.at[idx, f'ext_res_ystr_{iplane_ref}'] = v_res[0]
                calibrated_data.at[idx, f'ext_res_tsum_{iplane_ref}'] = v_res[1]
                calibrated_data.at[idx, f'ext_res_tdif_{iplane_ref}'] = v_res[2]
    
    
    # TimTrack result and residue plots ---------------------------------------
    if create_plots and residual_plots:
        timtrack_columns = ['x', 'xp', 't0', 'y', 'yp', 's']
        residual_columns = [
            'res_ystr_1', 'res_ystr_2', 'res_ystr_3', 'res_ystr_4',
            'res_tsum_1', 'res_tsum_2', 'res_tsum_3', 'res_tsum_4',
            'res_tdif_1', 'res_tdif_2', 'res_tdif_3', 'res_tdif_4'
        ]
        
        # Combined plot for all types
        plot_histograms_and_gaussian(calibrated_data, timtrack_columns, "Combined TimTrack Results", figure_number=1)
        plot_histograms_and_gaussian(calibrated_data, residual_columns, "Combined Residuals with Gaussian", figure_number=2, fit_gaussian=True, quantile=0.99)
        
        # Individual plots for each unique type
        unique_types = calibrated_data['type'].unique()
        for t in unique_types:
            subset_data = calibrated_data[calibrated_data['type'] == t]
            
            # Plot for the 'timtrack_columns' and 'residual_columns' based on type
            plot_histograms_and_gaussian(subset_data, timtrack_columns, f"TimTrack Results for Type {t}", figure_number=1)
            plot_histograms_and_gaussian(subset_data, residual_columns, f"Residuals with Gaussian for Type {t}", figure_number=2, fit_gaussian=True, quantile=0.99)
    # -------------------------------------------------------------------------
    
    
    # FILTER 6: TSUM, TDIF, QSUM, QDIF TIMTRACK X, Y, etc. FILTER --> IF THE
    #   RESULT IS OUT OF RANGE, REMOVE THE MODULE WITH LARGEST RESIDUE
    for index, row in calibrated_data.iterrows():
        # Check if x, y, or t0 is outside the desired range
        if (row['t0'] > t0_right_filter or row['t0'] < t0_left_filter) or \
            (row['x'] > pos_filter or row['x'] < -pos_filter or row['x'] == 0) or \
            (row['y'] > pos_filter or row['y'] < -pos_filter or row['y'] == 0) or \
            (row['xp'] > proj_filter or row['xp'] < -proj_filter or row['xp'] == 0) or \
            (row['yp'] > proj_filter or row['yp'] < -proj_filter or row['yp'] == 0) or \
            (row['s'] > slowness_filter_right or row['s'] < slowness_filter_left or row['s'] == 0) or\
            (row['charge_event'] > charge_event_right_filter or row['charge_event'] < charge_event_left_filter or row['charge_event'] == 0):

            # Find the module with the largest absolute residue value
            max_residue = 0
            module_to_zero = None
            
            for i in range(1, 5):
                if res_ana_removing_planes:
                    res_tsum = abs(row[f'ext_res_tsum_{i}'])
                    res_tdif = abs(row[f'ext_res_tdif_{i}'])
                    res_ystr = abs(row[f'ext_res_ystr_{i}'])
                else:
                    res_tsum = abs(row[f'res_tsum_{i}'])
                    res_tdif = abs(row[f'res_tdif_{i}'])
                    res_ystr = abs(row[f'res_ystr_{i}'])
                
                # Calculate the maximum residue for the module
                max_module_residue = max(res_tsum, res_tdif, res_ystr)
                
                if max_module_residue > max_residue:
                    max_residue = max_module_residue
                    module_to_zero = i
    
            # If a module is identified, set related values to 0
            if module_to_zero:
                calibrated_data.at[index, f'Y_{module_to_zero}'] = 0
                calibrated_data.at[index, f'P{module_to_zero}_T_sum_final'] = 0
                calibrated_data.at[index, f'P{module_to_zero}_T_diff_final'] = 0
                calibrated_data.at[index, f'P{module_to_zero}_Q_sum_final'] = 0
    
    # FILTER 7: TSUM, TDIF, QSUM, QDIF TIMTRACK RESIDUE FILTER --> 0 THE COMPONENT THAT HAS LARGE RESIDUE
    for index, row in calibrated_data.iterrows():
        for i in range(1, 5):
            if res_ana_removing_planes:
                if abs(row[f'ext_res_tsum_{i}']) > ext_res_tsum_filter or \
                    abs(row[f'ext_res_tdif_{i}']) > ext_res_tdif_filter or \
                    abs(row[f'ext_res_ystr_{i}']) > ext_res_ystr_filter:
                    
                    calibrated_data.at[index, f'Y_{i}'] = 0
                    calibrated_data.at[index, f'P{i}_T_sum_final'] = 0
                    calibrated_data.at[index, f'P{i}_T_diff_final'] = 0
                    calibrated_data.at[index, f'P{i}_Q_sum_final'] = 0
            else:
                if abs(row[f'res_tsum_{i}']) > res_tsum_filter or \
                    abs(row[f'res_tdif_{i}']) > res_tdif_filter or \
                    abs(row[f'res_ystr_{i}']) > res_ystr_filter:
                    
                    calibrated_data.at[index, f'Y_{i}'] = 0
                    calibrated_data.at[index, f'P{i}_T_sum_final'] = 0
                    calibrated_data.at[index, f'P{i}_T_diff_final'] = 0
                    calibrated_data.at[index, f'P{i}_Q_sum_final'] = 0
                    
    print("-----------------------------------------")
    four_planes = len(calibrated_data[calibrated_data.type == 1234])
    print(f"Events that are 1234: {four_planes}")
    print(f"Events that are 123: {len(calibrated_data[calibrated_data.type == 123])}")
    print(f"Events that are 234: {len(calibrated_data[calibrated_data.type == 234])}")
    
    planes134 = len(calibrated_data[calibrated_data.type == 134])
    print(f"Events that are 134: {planes134}")
    planes124 = len(calibrated_data[calibrated_data.type == 124])
    print(f"Events that are 124: {planes124}")
    
    eff_2 = 1 - (planes134) / (four_planes + planes134 + planes124)
    print(f"First estimate of eff_2 ={eff_2}")
    
    eff_3 = 1 - (planes124) / (four_planes + planes134 + planes124)
    print(f"First estimate of eff_3 ={eff_3}")
    
    iteration += 1
    # End of TimTrack loop ---------------------------------------------------------------

def calculate_angles(xproj, yproj):
    phi = np.arctan2(yproj, xproj)
    theta = np.arccos(1 / np.sqrt(xproj**2 + yproj**2 + 1))
    return theta, phi

theta, phi = calculate_angles(calibrated_data['xp'], calibrated_data['yp'])
new_columns_df = pd.DataFrame({'theta': theta, 'phi': phi}, index=calibrated_data.index)
calibrated_data = pd.concat([calibrated_data, new_columns_df], axis=1)
calibrated_data = calibrated_data.copy()


if create_plots:
    # Scatter plot of alt_theta vs alt_phi
    plt.figure(figsize=(8, 6))
    plt.scatter(calibrated_data['phi'], calibrated_data['theta'], alpha=0.7, s = 1)
    plt.xlabel('Azimuth (φ) [radians]')
    plt.ylabel('Zenith (θ) [radians]')
    plt.title('Scatter Plot of Fitted Angles')
    plt.grid(True)
    plt.show()
    
    if save_plots:
        name_of_file = 'timtrack_fitting_results_scatter_combination_projections'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')

    # Show plot if enabled
    if show_plots:
        plt.show()

    plt.close()

# FILTER 8: X, Y, T FILTER: IGNORE THE EVENT IF (AFTER ALL) ANY OF THEM GOES OUR OF BOUNDS
for col in calibrated_data.columns:
    if 't0' == col:
        calibrated_data.loc[:, col] = np.where(
            (calibrated_data[col] > t0_right_filter) | (calibrated_data[col] < t0_left_filter), 0, calibrated_data[col])
    if 'x' == col or 'y' == col:
        cond_bound = (calibrated_data[col] > pos_filter) | (calibrated_data[col] < -pos_filter)
        cond_zero = (calibrated_data[col] == 0)
        calibrated_data.loc[:, col] = np.where((cond_bound | cond_zero), 0, calibrated_data[col])
    if 'xp' == col or 'yp' == col:
        cond_bound = (calibrated_data[col] > proj_filter) | (calibrated_data[col] < -proj_filter)
        cond_zero = (calibrated_data[col] == 0)
        calibrated_data.loc[:, col] = np.where((cond_bound | cond_zero), 0, calibrated_data[col])
    if 's' == col:
        cond_bound = (calibrated_data[col] > slowness_filter_right) | (calibrated_data[col] < slowness_filter_left)
        cond_zero = (calibrated_data[col] == 0)
        calibrated_data.loc[:, col] = np.where((cond_bound | cond_zero), 0, calibrated_data[col])
    
cond = (calibrated_data['x'] != 0) & (calibrated_data['xp'] != 0) &\
       (calibrated_data['y'] != 0) & (calibrated_data['yp'] != 0) &\
       (calibrated_data['s'] != 0) & (calibrated_data['t0'] != 0)

for col in calibrated_data.columns:
    if 'alt_x' == col or 'alt_y' == col:
        cond_bound = (calibrated_data[col] > pos_filter) | (calibrated_data[col] < -pos_filter)
        cond_zero = (calibrated_data[col] == 0)
        calibrated_data.loc[:, col] = np.where((cond_bound | cond_zero), 0, calibrated_data[col])

# print(calibrated_data.columns)
final_data = calibrated_data.loc[cond].copy()

print("----------------------------------------------------------------------")
print("----------------------- Calculating some stuff -----------------------")
print("----------------------------------------------------------------------")

df_plot_ancillary = final_data.copy()

# Keep the rows where charge_event is between 0 and 50

cond = ( df_plot_ancillary['charge_1'] < 250 ) &\
    ( df_plot_ancillary['charge_2'] < 250 ) &\
    ( df_plot_ancillary['charge_3'] < 250 ) &\
    ( df_plot_ancillary['charge_4'] < 250 )

df_plot_ancillary = df_plot_ancillary.loc[cond].copy()
df_plot_ancillary = df_plot_ancillary[(df_plot_ancillary['charge_event'] > 0) & (df_plot_ancillary['charge_event'] < 600)]

if create_plots:
    from sklearn.mixture import GaussianMixture

    # Define number of Gaussian components (can be changed)
    num_components = 3  # Modify this to change the number of Gaussian components
    
    df_plot = df_plot_ancillary.copy()
    # Select the number of planes involved
    # df_plot = df_plot[df_plot['type'].astype(int) >= 100].copy()

    # Fit Gaussian Mixture Model (GMM) with specified components for xp and yp
    gmm_xp = GaussianMixture(n_components=num_components)
    gmm_xp.fit(df_plot[['xp']])

    gmm_yp = GaussianMixture(n_components=num_components)
    gmm_yp.fit(df_plot[['yp']])

    # Extract standard deviations of the fitted Gaussians from the GMM
    xp_std_devs = np.sqrt(gmm_xp.covariances_).flatten()
    yp_std_devs = np.sqrt(gmm_yp.covariances_).flatten()

    # Generate data for visualization
    x_vals = np.linspace(df_plot['xp'].min(), df_plot['xp'].max(), 1000)
    y_vals = np.linspace(df_plot['yp'].min(), df_plot['yp'].max(), 1000)

    xp_gmm_pdf = np.exp(gmm_xp.score_samples(x_vals.reshape(-1, 1)))
    yp_gmm_pdf = np.exp(gmm_yp.score_samples(y_vals.reshape(-1, 1)))

    # Define plotting parameters
    columns_of_interest = ['phi', 'xp', 'yp', 'theta', 'charge_event']
    num_bins = 100
    fig, axes = plt.subplots(len(columns_of_interest), len(columns_of_interest), figsize=(15, 15))

    for i in range(len(columns_of_interest)):
        for j in range(len(columns_of_interest)):
            ax = axes[i, j]
            if i < j:
                ax.axis('off')  # Leave the lower triangle blank
            elif i == j:
                # Diagonal: 1D histogram with independent axes
                hist_data = df_plot[columns_of_interest[i]]
                hist, bins = np.histogram(hist_data, bins=num_bins, density=True)
                bin_centers = 0.5 * (bins[1:] + bins[:-1])
                cmap = plt.get_cmap('turbo')

                for k in range(len(hist)):
                    ax.bar(bin_centers[k], hist[k], width=bins[1] - bins[0], color=cmap(hist[k] / max(hist)))

                # Overlay GMM fit with std devs in the legend
                if columns_of_interest[i] == 'xp':
                    ax.plot(x_vals, xp_gmm_pdf, 'r-', label=f"Std devs: {', '.join([f'{std:.2f}' for std in xp_std_devs])}")
                    ax.legend(fontsize=8)

                elif columns_of_interest[i] == 'yp':
                    ax.plot(y_vals, yp_gmm_pdf, 'r-', label=f"Std devs: {', '.join([f'{std:.2f}' for std in yp_std_devs])}")
                    ax.legend(fontsize=8)

                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Upper triangle: hexbin plots
                x_data = df_plot[columns_of_interest[j]]
                y_data = df_plot[columns_of_interest[i]]
                hb = ax.hexbin(x_data, y_data, gridsize=num_bins, cmap='turbo')

            if i != len(columns_of_interest) - 1:
                ax.set_xticklabels([])  # Remove x-axis labels except for the last row
            if j != 0:
                ax.set_yticklabels([])  # Remove y-axis labels except for the first column
            if i == len(columns_of_interest) - 1:  # Last row, set x-labels
                ax.set_xlabel(columns_of_interest[j])
            if j == 0:  # First column, set y-labels
                ax.set_ylabel(columns_of_interest[i])

    # Print fitted Gaussian parameters
    print("--------------------------------")
    for i in range(num_components):
        print(f"XP Gaussian {i+1} std: {xp_std_devs[i]:.2f}")
    for i in range(num_components):
        print(f"YP Gaussian {i+1} std: {yp_std_devs[i]:.2f}")
    print("--------------------------------")

    # Adjust layout and title
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.suptitle(f"Fitted with {num_components}-Component GMM Fit", fontsize=16)

    # Save plot if enabled
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


# --------------------------------------------------------------

def plot_hexbin_matrix(df, columns_of_interest, filter_conditions, title, save_plots, show_plots, base_directories, fig_idx, plot_list, num_bins=60):
    """
    Generates a hexbin matrix plot with histograms on the diagonal.
    
    Parameters:
    - df: Pandas DataFrame containing the data
    - columns_of_interest: List of column names to include in the plot
    - filter_conditions: List of tuples (column, min_value, max_value) to filter df
    - title: Title of the plot
    - num_bins: Number of bins for histograms and hexbin plots (default: 60)
    - save_plots: Boolean to save the plot (default: False)
    - show_plots: Boolean to display the plot (default: True)
    - base_directory: Path to save the plot if save_plots is True
    - fig_idx: Index to differentiate saved plot filenames
    - plot_list: List to store the saved plot filenames
    """
    
    # 'x', 'y', 'theta', 'phi', 'xp', 'yp', 'charge_event'
    
    axis_limits = {
        'x': [-pos_filter, pos_filter],
        'y': [-pos_filter, pos_filter],
        'alt_x': [-pos_filter, pos_filter],
        'alt_y': [-pos_filter, pos_filter],
        'theta': [0, 1.3],
        'phi': [-np.pi, np.pi],
        'alt_theta': [0, 1.3],
        'alt_phi': [-np.pi, np.pi],
        'xp': [-2, 2],
        'yp': [-2, 2],
        'charge_event': [0, 600],
        'charge_1': [0, 250],
        'charge_2': [0, 250],
        'charge_3': [0, 250],
        'charge_4': [0, 250],
        's': [slowness_filter_left, slowness_filter_right],
        'alt_slowness': [slowness_filter_left, slowness_filter_right],
        'th_chi': [0, 0.03]
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

                # Apply determined limits
                ax.set_xlim(auto_limits[x_col])
                
                # If the column is 'charge_1, 2, 3 or 4', set logscale in Y
                if x_col.startswith('charge'):
                    ax.set_yscale('log')
                
            else:
                # Upper triangle: hexbin plots
                x_data = df[x_col]
                y_data = df[y_col]
                # Remove zeroes and nans
                cond = (x_data != 0) & (y_data != 0) & (~np.isnan(x_data)) & (~np.isnan(y_data))
                x_data = x_data[cond]
                y_data = y_data[cond]
                ax.hexbin(x_data, y_data, gridsize=num_bins, cmap='turbo')
                ax.set_facecolor(plt.cm.turbo(0))
                
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
    
    # Save plot if enabled
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


df_cases_2 = [
    ([("type", 12, 12)], "1-2 cases"),
    ([("type", 23, 23)], "2-3 cases"),
    ([("type", 34, 34)], "3-4 cases"),
    ([("type", 13, 13)], "1-3 cases"),
    ([("type", 14, 14)], "1-4 cases"),
    ([("type", 123, 123)], "1-2-3 cases"),
    ([("type", 234, 234)], "2-3-4 cases"),
    ([("type", 124, 124)], "1-2-4 cases"),
    ([("type", 134, 134)], "1-3-4 cases"),
    ([("type", 1234, 1234)], "1-2-3-4 cases"),
]

# for filters, title in df_cases_2:
#     fig_idx = plot_hexbin_matrix(
#         df_plot_ancillary,
#         ['x', 'y', 'theta', 'phi', 'xp', 'yp', 'charge_event'],
#         filters,
#         title,
#         save_plots,
#         show_plots,
#         base_directories,
#         fig_idx,
#         plot_list
#     )


# for filters, title in df_cases_2:
#     # Extract the relevant charge numbers from the title (e.g., "1-2 cases" -> [1, 2])
#     relevant_charges = [f"charge_{n}" for n in map(int, title.split()[0].split('-'))]

#     # Define the columns of interest dynamically
#     columns_of_interest = ['x', 'y', 'theta', 'phi', 'xp', 'yp'] + relevant_charges

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


# for filters, title in df_cases_2:
#     fig_idx = plot_hexbin_matrix(
#         df_plot_ancillary,
#         ['x', 'y', 'theta', 'phi', 'xp', 'yp', 's', 'th_chi'],
#         filters,
#         title,
#         save_plots,
#         show_plots,
#         base_directories,
#         fig_idx,
#         plot_list
#     )


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

# for filters, title in df_cases_2:
#     fig_idx = plot_hexbin_matrix(
#         df_plot_ancillary,
#         ['theta', 'phi', 'alt_theta', 'alt_phi'],
#         filters,
#         title,
#         save_plots,
#         show_plots,
#         base_directories,
#         fig_idx,
#         plot_list
#     )

# for filters, title in df_cases_2:
#     fig_idx = plot_hexbin_matrix(
#         df_plot_ancillary,
#         ['x', 'y', 'alt_x', 'alt_y'],
#         filters,
#         title,
#         save_plots,
#         show_plots,
#         base_directories,
#         fig_idx,
#         plot_list
#     )

# for filters, title in df_cases_2:
#     fig_idx = plot_hexbin_matrix(
#         df_plot_ancillary,
#         ['alt_x', 'alt_y', 'alt_theta', 'alt_phi'],
#         filters,
#         title,
#         save_plots,
#         show_plots,
#         base_directories,
#         fig_idx,
#         plot_list
#     )

for filters, title in df_cases_2:
    fig_idx = plot_hexbin_matrix(
        df_plot_ancillary,
        ['alt_slowness', 's', 'theta', 'phi', 'alt_theta', 'alt_phi'],
        filters,
        title,
        save_plots,
        show_plots,
        base_directories,
        fig_idx,
        plot_list
    )

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# Including th_chi filtered -------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

chi_histo = True
if chi_histo and create_plots:
    data = df_plot_ancillary['th_chi'].dropna()  # Remove NaN values if any
    data = data[data != 0]
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=300, alpha=0.5, color='blue', log=True)
    plt.title('Histogram of th_chi', fontsize=16)
    plt.xlabel('th_chi', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    
    if save_plots:
        name_of_file = 'th_chi_histo'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: plt.show()
    plt.close()

df_plot_ancillary = df_plot_ancillary[(df_plot_ancillary['th_chi'] > 0.0001) & (df_plot_ancillary['th_chi'] < 0.02)]

if chi_histo and create_plots:
    data = df_plot_ancillary['th_chi'].dropna()  # Remove NaN values if any
    data = data[data != 0]
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=300, alpha=0.5, color='blue', log=True)
    plt.title('Histogram of th_chi', fontsize=16)
    plt.xlabel('th_chi', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_plots:
        name_of_file = 'th_chi_histo_filt'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: plt.show()
    plt.close()


df_plot_ancillary = df_plot_ancillary[df_plot_ancillary['type'].astype(int) >= 1000].copy()
df_plot_ancillary = df_plot_ancillary[(df_plot_ancillary['th_chi'] > 0.00125) & (df_plot_ancillary['th_chi'] < 0.02)]

if chi_histo and create_plots:
    data = df_plot_ancillary['th_chi'].dropna()  # Remove NaN values if any
    data = data[data != 0]
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=300, alpha=0.5, color='blue', log=True)
    plt.title('Histogram of th_chi, only four planes', fontsize=16)
    plt.xlabel('th_chi', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_plots:
        name_of_file = 'th_chi_histo_filt_four_planes'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    
    if show_plots: plt.show()
    plt.close()


# if create_plots or create_essential_plots:
#     df_plot = df_plot_ancillary[df_plot_ancillary['type'].astype(int) >= 1000].copy()
    
#     columns_of_interest = ['x', 'y', 'theta', 'phi', 't0', 's', 'th_chi', 'charge_event']
#     num_bins = 30
#     fig, axes = plt.subplots(8, 8, figsize=(15, 15))
#     for i in range(8):
#         for j in range(8):
#             ax = axes[i, j]
#             if i < j:
#                 ax.axis('off')  # Leave the lower triangle blank
#             elif i == j:
#                 # Diagonal: 1D histogram with independent axes
#                 hist_data = df_plot[columns_of_interest[i]]
#                 hist, bins = np.histogram(hist_data, bins=num_bins)
#                 bin_centers = 0.5 * (bins[1:] + bins[:-1])
#                 norm = plt.Normalize(hist.min(), hist.max())
#                 cmap = plt.get_cmap('turbo')
#                 for k in range(len(hist)):
#                     ax.bar(bin_centers[k], hist[k], width=bins[1] - bins[0], color=cmap(norm(hist[k])))
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#             else:
#                 # Upper triangle: hexbin plots
#                 x_data = df_plot[columns_of_interest[j]]
#                 y_data = df_plot[columns_of_interest[i]]
#                 hb = ax.hexbin(x_data, y_data, gridsize=num_bins, cmap='turbo')
#             if i != 6:
#                 ax.set_xticklabels([])  # Remove x-axis labels except for the last row
#             if j != 0:
#                 ax.set_yticklabels([])  # Remove y-axis labels except for the first column
#             if i == 7:  # Last row, set x-labels
#                 ax.set_xlabel(columns_of_interest[j])
#             if j == 0:  # First column, set y-labels
#                 ax.set_ylabel(columns_of_interest[i])
#     plt.subplots_adjust(wspace=0.05, hspace=0.05)
#     plt.suptitle("Only four planes, chisq filtered")
    
#     if save_plots:
#         name_of_file = 'timtrack_results_hexbin_combination_th_chi'
#         final_filename = f'{fig_idx}_{name_of_file}.png'
#         fig_idx += 1

#         save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
#         plot_list.append(save_fig_path)
#         plt.savefig(save_fig_path, format='png')
    
#     if show_plots: plt.show()
#     plt.close()


# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------


print("-----------------------------")
data_purity = len(final_data) / raw_data_len*100
print(f"Data purity is {data_purity:.1f}%")

global_variables['purity_of_data_percentage'] = data_purity

# ------------------------------------------------------------------------------------------------
# Statistical comprobation ----------------------------------------------------
# ------------------------------------------------------------------------------------------------

if create_plots or create_essential_plots:
    test_data = final_data.copy()
    
    test_data['datetime'] = pd.to_datetime(test_data['datetime'], errors='coerce')
    test_data = test_data.set_index('datetime')
    
    df_plot_1 = test_data.copy()
    df_plot_2 = test_data[test_data['type'].astype(int) >= 100].copy()
    df_plot_3 = test_data[test_data['type'].astype(int) >= 1000].copy()
    df_plot_4 = og_data.copy()  # Original dataset
    
    datasets = {'All Data': df_plot_1, 
                'Type >= 100': df_plot_2, 
                'Type >= 1000': df_plot_3, 
                'Original Data': df_plot_4}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()  # Flatten axes for easy iteration
    
    for ax, (name, df) in zip(axes, datasets.items()):
        df.index = pd.to_datetime(df.index, errors='coerce')  # Converts index to datetime
        events_per_second = df.index.floor('s').value_counts()
        # print(events_per_second)
        
        hist_data = events_per_second.value_counts().sort_index()
        lambda_estimate = events_per_second.mean()
        x_values = np.arange(0, hist_data.index.max() + 1)
        poisson_pmf = poisson.pmf(x_values, lambda_estimate)
        poisson_pmf_scaled = poisson_pmf * len(events_per_second)
        ax.bar(hist_data.index, hist_data.values, width=1, color='blue', alpha=0.5, label='Histogram')
        ax.plot(hist_data.index, hist_data.values, color='blue', alpha=1)
        ax.plot(x_values, poisson_pmf_scaled, 'r-', lw=2, label=f'Poisson fit ($\\lambda={lambda_estimate:.2f}$ cts/s)')
        ax.set_title(f'{name} - Histogram and Poisson Fit')
        ax.set_xlabel('Number of Events per Second')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle('Event Rate Histograms with Poisson Fits', fontsize=16, y=1.03)
    
    if save_plots:
        name_of_file = 'events_per_second_'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
        
    if show_plots:
        plt.show()


# ------------------------------------------------------------------------------------------------
# Time window plotting
# ------------------------------------------------------------------------------------------------

if create_plots:
    from matplotlib.cm import get_cmap

    cases = [1234, 123, 234, 124, 134, 12, 23, 34, 13, 14, 24]
    cmap = get_cmap('turbo')
    colors = cmap(np.linspace(0, 1, len(cases)))

    # Define window widths
    widths = np.linspace(1, 20, 80)
    plt.figure(figsize=(10, 6))

    for idx, case in enumerate(cases):
        data_case = final_data[final_data["type"] == case].copy()
        
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
    plt.tight_layout()
    
    if save_plots:
        name_of_file = 'window_coincidence_count_'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1

        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
        
    if show_plots:
        plt.show()

print("----------------------------------------------------------------------")
print("-------------------------- Save and finish ---------------------------")
print("----------------------------------------------------------------------")

# Put the global_variables as columns in the final_data dataframe
for key, value in global_variables.items():
    if key not in final_data.columns:
        print(f"Adding {key} to the dataframe.")
        final_data[key] = value
    else:
        print(f"Warning: Column '{key}' already exists in the DataFrame. Skipping addition.")

# Replace the TimTrack fitting angle results with the alternative ones
if alternative_fitting:
    if 'alt_theta' in final_data.columns:
        final_data['theta'] = final_data['alt_theta']
        final_data['phi'] = final_data['alt_phi']
        final_data['x'] = final_data['alt_x']
        final_data['y'] = final_data['alt_y']

# Round to 4 significant digits -----------------------------------------------
print("Rounding the dataframe values.")

def round_to_4_significant_digits(x):
    try:
        # Use builtins.float to avoid any overridden names
        return builtins.float(f"{builtins.float(x):.4g}")
    except (builtins.ValueError, builtins.TypeError):
        return x
    
for col in final_data.select_dtypes(include=[np.number]).columns:
    final_data.loc[:, col] = final_data[col].apply(round_to_4_significant_digits)


# ---------------------------------------------------------------------------------------------
# Change 'datetime' column to 'Time'
# print(final_data.columns)

if 'datetime' in final_data.columns:
    final_data.rename(columns={'datetime': 'Time'}, inplace=True)
else:
    print("Column 'datetime' not found in DataFrame!")
# ---------------------------------------------------------------------------------------------


# Save the data ---------------------------------------------------------------
if save_full_data: # Save a full version of the data, for different studies and debugging
    final_data.to_csv(save_full_path, index=False, sep=',', float_format='%.5g')
    print(f"Datafile saved in {save_full_filename}'.")

# Save a reduced version of the data always, to proceed with the analysis
columns_to_keep = [
    'Time', 'CRP_avg', 'x', 'y', 'theta', 'phi', 't0', 's', 'type', 'charge_event',
    'Q_P1s1', 'Q_P1s2', 'Q_P1s3', 'Q_P1s4',
    'Q_P2s1', 'Q_P2s2', 'Q_P2s3', 'Q_P2s4',
    'Q_P3s1', 'Q_P3s2', 'Q_P3s3', 'Q_P3s4',
    'Q_P4s1', 'Q_P4s2', 'Q_P4s3', 'Q_P4s4'
]

reduced_df = final_data[columns_to_keep]

reduced_df.to_csv(save_list_path, index=False, sep=',', float_format='%.5g')
print(f"Datafile saved in {save_filename}. Path is {save_list_path}")


# Save the calibrations -------------------------------------------------------
new_row = {'Time': start_time}

for i, module in enumerate(['P1', 'P2', 'P3', 'P4']):
    for j in range(4):
        strip = j + 1
        new_row[f'{module}_s{strip}_Q_sum'] = ( QF_pedestal[i][j] + QB_pedestal[i][j] ) / 2
        new_row[f'{module}_s{strip}_T_sum'] = calibration_times[i, j]
        # new_row[f'{module}_s{strip}_Q_dif'] = calibration_Q_FB[i, j]
        new_row[f'{module}_s{strip}_Q_dif'] = ( QF_pedestal[i][j] - QB_pedestal[i][j] ) / 2
        # new_row[f'{module}_s{strip}_T_dif'] = calibration_T[i, j]
        new_row[f'{module}_s{strip}_T_dif'] = Tdiff_cal[i][j]

if os.path.exists(csv_path):
    # Load the existing DataFrame
    calibrations_df = pd.read_csv(csv_path, parse_dates=['Time'])
else:
    columns = ['Time'] + [
        f'{module}_s{strip}_{var}'
        for module in ['P1', 'P2', 'P3', 'P4']
        for strip in range(1, 5)
        for var in ['Q_sum', 'T_sum', 'Q_dif', 'T_dif']
    ]
    calibrations_df = pd.DataFrame(columns=columns)

# Check if the current time already exists
existing_row_index = calibrations_df[calibrations_df['Time'] == start_time].index

if not existing_row_index.empty:
    # Update the existing row
    calibrations_df.loc[existing_row_index[0]] = new_row
    print(f"Updated existing calibration for date: {start_time}")
else:
    # Append the new row
    calibrations_df = pd.concat([calibrations_df, pd.DataFrame([new_row])], ignore_index=True)
    print(f"Added new calibration for date: {start_time}")

calibrations_df.sort_values(by='Time', inplace=True)
calibrations_df.to_csv(csv_path, index=False, float_format='%.5g')
print(f'{csv_path} updated with the calibrations for this folder.')     

# Create and save the PDF -----------------------------------------------------
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

# Move the original datafile to PROCESSED -------------------------------------
print("Moving file to COMPLETED directory...")
# shutil.move(file_path, completed_path)
shutil.move(file_path, completed_file_path)
print("************************************************************")
print(f"File moved from\n{file_path}\nto:\n{completed_file_path}")
print("************************************************************")

if os.path.exists(temp_file):
    print("Removing temporary file...")
    os.remove(temp_file)

# Store the current time at the end
end_execution_time_counting = datetime.now()
time_taken = (end_execution_time_counting - start_execution_time_counting).total_seconds() / 60
print(f"Time taken for the whole execution: {time_taken:.2f} minutes")

print("----------------------------------------------------------------------")
print("------------------- Finished list_events creation --------------------")
print("----------------------------------------------------------------------\n\n\n")