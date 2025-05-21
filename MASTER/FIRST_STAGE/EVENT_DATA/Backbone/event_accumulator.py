#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
columns_to_keep = [
    # Timestamp and identifiers
    'Time', 'original_tt', 'processed_tt', 'tracking_tt',

    # Summary metrics and quality flags
    'CRT_avg', 'sigmoid_width', 'background_slope',
    'one_side_events', 'purity_of_data_percentage',
    'unc_y', 'unc_tsum', 'unc_tdif',

    # Alternative reconstruction outputs
    'alt_x', 'alt_y', 'alt_theta', 'alt_phi', 'alt_s', 'alt_th_chi',

    # TimTrack reconstruction outputs
    'x', 'y', 'theta', 'phi', 's', 'th_chi',

    # Strip-level time and charge info (ordered by plane and strip)
    *[f'Q_P{p}s{s}' for p in range(1, 5) for s in range(1, 5)]
]
'''

import numpy as np
import pandas as pd
import sys
import os
import builtins
import shutil
import random
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.optimize import minimize

last_file_test = False
reanalyze_completed = True
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

# Check if the script has an argument
if len(sys.argv) < 2:
    print("Error: No station provided.")
    print("Usage: python3 script.py <station>")
    sys.exit(1)

# Get the station argument
station = sys.argv[1]
print(f"Station: {station}")

date_execution = datetime.now().strftime("%y-%m-%d_%H.%M.%S")


# -----------------------------------------------------------------------------
# -------------------------- Variables of execution ---------------------------
# -----------------------------------------------------------------------------

remove_outliers = True
create_plots = True
save_plots = True
create_pdf = True
force_replacement = True  # Creates a new datafile even if there is already one that looks complete
show_plots = False
caye_high_mid_limit_angle = 15

hans_high_mid_limit_angle = 10
hans_low_mid_limit_angle = 40

polya_fit = True
real_strip_case_study = False
multiplicity_calculations = True
crosstalk_probability = True
georgys = False

event_acc_global_variables = {
    "poisson_rejected": 0,
    "hans_reg_high_to_mid": hans_high_mid_limit_angle,
    "hans_reg_mid_to_low": hans_low_mid_limit_angle,
    "caye_reg_high_to_mid": caye_high_mid_limit_angle,
}

event_acc_global_variables["poisson_rejected"] = 0


print("----------------------------------------------------------------------")
print("--------------------- Starting the directories -----------------------")
print("----------------------------------------------------------------------")

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
    "full_acc_events_directory": os.path.join(working_directory, "FULL_ACC_EVENTS_DIRECTORY"),
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

source_dir = base_directories["acc_events_directory"]
rejected_dir = base_directories["acc_rejected_directory"]

for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)
    
    # Check if it's a file
    if os.path.isfile(file_path):
        # Count the number of lines in the file
        with open(file_path, "r") as f:
            line_count = sum(1 for _ in f)

        # Move the file if it has < 15 or > 100 rows
        if line_count < 15 or line_count > 300:
            shutil.move(file_path, os.path.join(rejected_dir, filename))
            print(f"Moved: {filename}")


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

def classify_region(row):
    phi = row['phi'] * 180 / np.pi  + row['phi_north'] # Convert phi to degrees
    theta = row['theta'] * 180 / np.pi
    if 0 <= theta < caye_high_mid_limit_angle:
        return 'High'
    elif caye_high_mid_limit_angle <= theta <= 90:
        if -45 <= phi < 45:
            return 'N'
        elif 45 <= phi < 135:
            return 'E'
        elif -135 <= phi < -45:
            return 'W'
        else:
            return 'S'


# Hans' angular map division
high_regions_hans = ['V']
mid_regions_hans = ['N.M', 'NE.M', 'E.M', 'SE.M', 'S.M', 'SW.M', 'W.M', 'NW.M']
low_regions_hans = ['N.H', 'E.H', 'S.H', 'W.H']

def classify_region_hans(row):
    phi = row['phi'] * 180/np.pi
    theta = row['theta'] * 180/np.pi
    
    # if int(row['type']) <= 100:
    #     return 'None'
    
    if 0 <= theta < hans_high_mid_limit_angle:
        return 'V'
    elif hans_high_mid_limit_angle <= theta <= hans_low_mid_limit_angle:
        if -22.5 <= phi < 22.5:
            return 'N.M'
        elif 22.5 <= phi < 67.5:
            return 'NE.M'
        elif 67.5 <= phi < 112.5:
            return 'E.M'
        elif 112.5 <= phi < 157.5:
            return 'SE.M'
        elif -180 <= phi < -157.5 or 157.5 <= phi <= 180:
            return 'S.M'
        elif -157.5 <= phi < -112.5:
            return 'SW.M'
        elif -112.5 <= phi < -67.5:
            return 'W.M'
        else:  # -67.5 <= phi < -22.5
            return 'NW.M'
    elif 40 < theta <= 90:
        if -45 <= phi < 45:
            return 'N.H'
        elif 45 <= phi < 135:
            return 'E.H'
        elif -135 <= phi < -45:
            return 'W.H'
        else:  # phi >= 135 or phi < -135
            return 'S.H'

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
df = pd.read_csv(file_path, sep=',')
df['Time'] = pd.to_datetime(df['Time'], errors='coerce') # Added errors='coerce' to handle NaT values
print(f"Number of events in the file: {len(df)}")

min_time_original = df['Time'].min()
max_time_original = df['Time'].max()
valid_times = df['Time'].dropna()

if not valid_times.empty:
    min_time_valid = valid_times.min()
    
    # Check if the min value with NaT differs from the valid min value
    if min_time_original != min_time_valid:
        print("Notice: The minimum value from 'Time' column differs from the smallest valid datetime.")
        print("Original min value (including NaT):", min_time_original)
        print("Valid min value (ignoring NaT):", min_time_valid)
    
    first_datetime = min_time_valid
    filename_save_suffix = first_datetime.strftime('%y-%m-%d_%H.%M.%S')
else:
    # first_datetime = None
    sys.exit("No valid datetime values found in the 'Time' column. Exiting...")

print("Filename save suffix:", filename_save_suffix)

full_save_filename = f"full_accumulated_events_{filename_save_suffix}.csv"
full_save_path = os.path.join(base_directories["full_acc_events_directory"], full_save_filename)

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
    start_time = min_time_original
    end_time = max_time_original
    
    # Read and preprocess the input file only once
    input_file["start"] = pd.to_datetime(input_file["start"], format="%Y-%m-%d", errors="coerce")
    input_file["end"] = pd.to_datetime(input_file["end"], format="%Y-%m-%d", errors="coerce")
    input_file["end"].fillna(pd.to_datetime('now'), inplace=True)

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

# Adjust z_positions and print
z_positions = z_positions - z_positions[0]
print(f"Z positions: {z_positions}")

z1 = z_positions[0]
z2 = z_positions[1]
z3 = z_positions[2]
z4 = z_positions[3]

print(df.columns.to_list())

# Rename every column that contains Q_M... for  Q_P...
for col in df.columns:
    if "Q_M" in col:
        new_col = col.replace("Q_M", "Q_P")
        df.rename(columns={col: new_col}, inplace=True)

print(df.columns.to_list())


print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("----------------------- Starting the analysis ------------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

print("----------------------------------------------------------------------")
print("----------------------------- Polya fit ------------------------------")
print("----------------------------------------------------------------------")

if polya_fit:
    print("Polya fit. WIP.")

    remove_crosstalk = True
    remove_streamer = True
    crosstalk_limit = 1
    streamer_limit = 900

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
    from scipy.interpolate import CubicSpline
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

    # merged_df = df.copy()

    if remove_crosstalk or remove_streamer:
        if remove_crosstalk:
            for col in merged_df.columns:
                if "Q_" in col and "s" in col:
                    merged_df[col] = merged_df[col].apply(lambda x: 0 if x < crosstalk_limit else x)
                        
        if remove_streamer:
            for col in merged_df.columns:
                if "Q_" in col and "s" in col:
                    merged_df[col] = merged_df[col].apply(lambda x: 0 if x > streamer_limit else x)     

    columns_to_drop = ['Time', 'CRT_avg', 'x', 'y', 'theta', 'phi', 's']
    merged_df = merged_df.drop(columns=columns_to_drop)

    # For all the columns apply the calibration and not change the name of the columns
    for col in merged_df.columns:
        merged_df[col] = interpolate_fast_charge(merged_df[col])

    # For each module, calculate the total charge per event, then store them in a dataframe
    total_charge = pd.DataFrame()
    for i in range(1, 5):
        total_charge[f"Q_P{i}"] = merged_df[[f"Q_P{i}s{j}" for j in range(1, 5)]].sum(axis=1)


    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import gamma
    from scipy.optimize import curve_fit

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
        counts, bin_edges = np.histogram(data, bins=100, range=(0, 1.1e7))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        bin_center = bin_centers[counts >= 0.05 * max(counts)][0]
        mask = (bin_centers > bin_center) & (counts > 0)
        x_fit = bin_centers[mask]
        y_fit = counts[mask]

        # Fit
        p0 = [1, 1e6, 0.6, max(counts), 1e6]
        bounds = ([0, 0, 0, 0, -1e16], [100, 1e16, 1, 1e16, 1e16])
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
            "nbar/alpha": nbar_fit / alpha_fit
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
    print(df_polya_fit)


print("----------------------------------------------------------------------")
print("-------------------- Real adjacent and single cases ------------------")
print("----------------------------------------------------------------------")

if real_strip_case_study:
    print("Real strip case study. WIP.")

    # Read and concatenate all files
    df_list = [df]  # Adjust delimiter if needed
    merged_df = pd.concat(df_list, ignore_index=True)

    # Drop duplicates if necessary
    merged_df.drop_duplicates(inplace=True)

    # Print the column names
    print(merged_df.columns.to_list())

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

    from scipy.interpolate import CubicSpline

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

    columns_to_drop = ['Time', 'CRT_avg', 'x', 'y', 'theta', 'phi', 's']
    merged_df = merged_df.drop(columns=columns_to_drop)

    print(merged_df.columns.to_list())

    # For all the columns apply the calibration and not change the name of the columns
    for col in merged_df.columns:
        merged_df[col] = interpolate_fast_charge(merged_df[col])


    # Initialize dictionaries to store charge distributions
    singles = {f'single_P{i}_s{j}': [] for i in range(1, 5) for j in range(1, 5)}
    double_adj = {f'double_P{i}_s{j}{j+1}': [] for i in range(1, 5) for j in range(1, 4)}
    double_non_adj = {f'double_P{i}_s{pair[0]}{pair[1]}': [] for i in range(1, 5) for pair in [(1,3), (2,4), (1,4)]}
    triple_adj = {f'triple_P{i}_s{j}{j+1}{j+2}': [] for i in range(1, 5) for j in range(1, 3)}
    triple_non_adj = {f'triple_P{i}_s{triplet[0]}{triplet[1]}{triplet[2]}': [] for i in range(1, 5) for triplet in [(1,2,4), (1,3,4)]}
    quadruples = {f'quadruple_P{i}_s1234': [] for i in range(1, 5)}

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

            # Double adjacent
            elif count == 2 and nonzero_strips[1] - nonzero_strips[0] == 1:
                key = f'double_P{i}_s{nonzero_strips[0]}{nonzero_strips[1]}'
                double_adj[key].append(tuple(charges))

            # Double non-adjacent
            elif count == 2 and nonzero_strips[1] - nonzero_strips[0] > 1:
                key = f'double_P{i}_s{nonzero_strips[0]}{nonzero_strips[1]}'
                if key in double_non_adj:
                    double_non_adj[key].append(tuple(charges))

            # Triple adjacent
            elif count == 3 and (nonzero_strips[2] - nonzero_strips[0] == 2):
                key = f'triple_P{i}_s{nonzero_strips[0]}{nonzero_strips[1]}{nonzero_strips[2]}'
                triple_adj[key].append(tuple(charges))

            # Triple non-adjacent
            elif count == 3 and (nonzero_strips[2] - nonzero_strips[0] > 2):
                key = f'triple_P{i}_s{nonzero_strips[0]}{nonzero_strips[1]}{nonzero_strips[2]}'
                if key in triple_non_adj:
                    triple_non_adj[key].append(tuple(charges))

            # Quadruple detection
            elif count == 4:
                key = f'quadruple_P{i}_s1234'
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

            combined_df = pd.concat([real_multiplicities[key] for key in matching_keys], axis=1)
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
                axs[a][i].hist(combined_df[column], bins=70, range=(0, 1500), histtype="step", linewidth=1.5, density=False)
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
            combined_df = pd.concat([combined_multiplicities[k] for k in matching_keys], axis=1)
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
                axs[row_idx][col_idx].hist(
                    combined_df[column_name],
                    bins=70,
                    range=(0, 2000),
                    histtype="step",
                    linewidth=1.5,
                    density=False
                )
                axs[row_idx][col_idx].set_title(f"{module} - {column_name}")
                axs[row_idx][col_idx].set_xlabel("Charge")
                axs[row_idx][col_idx].set_ylabel("Frequency")
                axs[row_idx][col_idx].grid(True)

            # Hide any unused subplots in this row
            for hidden_col_idx in range(col_idx + 1, max_columns):
                axs[row_idx][hidden_col_idx].axis("off")

        plt.tight_layout()
        figure_name = f"/{case}_sum.png"
        plt.savefig(figure_save_path + figure_name, dpi=150)
        plt.show()  # If you want interactive displays instead
        plt.close()

else:
    print("Real strip case study not available yet. WIP.")


print("----------------------------------------------------------------------")
print("----------------------- Multiplicity calculations --------------------")
print("----------------------------------------------------------------------")

if multiplicity_calculations:
    print("Multiplicity calculations. WIP.")

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

    remove_crosstalk = True
    crosstalk_limit = 3.5 #2.6

    remove_streamer = True
    streamer_limit = 90

    # Read and concatenate all files
    df_list = [df]  # Adjust delimiter if needed
    merged_df = pd.concat(df_list, ignore_index=True)

    # Drop duplicates if necessary
    merged_df.drop_duplicates(inplace=True)

    # Print the column names
    print(merged_df.columns.to_list())

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

    from scipy.interpolate import CubicSpline

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


    if remove_crosstalk or remove_streamer:
        if remove_crosstalk:
                for col in merged_df.columns:
                    if "Q_" in col and "s" in col:
                            merged_df[col] = merged_df[col].apply(lambda x: 0 if x < crosstalk_limit else x)
                    
        if remove_streamer:
                for col in merged_df.columns:
                    if "Q_" in col and "s" in col:
                            merged_df[col] = merged_df[col].apply(lambda x: 0 if x > streamer_limit else x)     


    columns_to_drop = ['Time', 'CRT_avg', 'x', 'y', 'theta', 'phi', 's']
    merged_df = merged_df.drop(columns=columns_to_drop)

    print(merged_df.columns.to_list())

    # For all the columns apply the calibration and not change the name of the columns
    for col in merged_df.columns:
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
                axs[i-1, j-1].hist(v, bins=200, range=(0, 1500))
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

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import nnls

    # Parameters
    selected_alpha = 0.7
    bin_number = 250 # 250
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

    # Step 1: Precompute 1–number_of_particles_bound_up single sums for each module
    hist_basis_all_modules = []  # [ [H1, H2, ..., H6] for each module ]

    number_of_particles_bound_up = 12

    for single_data in singles:
        single_data = np.array(single_data)
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

    import pandas as pd

    coeff_tables = {dt: pd.DataFrame(index=[f"S{n}" for n in range(1, number_of_particles_bound_up + 1)],
                                    columns=["M1", "M2", "M3", "M4"])
                    for dt in detection_types}

    # Accumulate event-weighted contributions per module
    component_counts = {
        "M1": np.zeros(number_of_particles_bound_up),
        "M2": np.zeros(number_of_particles_bound_up),
        "M3": np.zeros(number_of_particles_bound_up),
        "M4": np.zeros(number_of_particles_bound_up)
    }


    for i, (detection_type, df_group) in enumerate(zip(detection_types, df_data)):
        ax_hist = axes[i, 0]   # Left column: histograms and fit
        ax_scatter = axes[i, 1]  # Right column: scatter plot

        for j, (df_in, color, module) in enumerate(zip(df_group, module_colors, ['M1', 'M2', 'M3', 'M4'])):
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

            # Scatter: frequency of singles (model) vs multiple (data)
            ax_scatter.scatter(model, counts_df, label=module, color=color, s=1)
            min_val = max(np.min(model[model > 0]), np.min(counts_df[counts_df > 0]))
            max_val = max(np.max(model), np.max(counts_df))
            ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='y = x' if j == 0 else None)

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


    # Normalize the columns to the sum of each column
    coeff_tables_normalized = coeff_tables.copy()
    for detection_type, df_coeffs in coeff_tables.items():
        coeff_tables_normalized[detection_type] = df_coeffs.div(df_coeffs.sum(axis=0), axis=1)

    for detection_type, df_coeffs in coeff_tables_normalized.items():
        df_coeffs = df_coeffs.astype(float)
        df_percent = (df_coeffs * 100).round(1)
        print(f"\n===== Coefficients for {detection_type} (in %) =====")
        print(df_percent.to_string())  # Forces output in all environments


    import matplotlib.pyplot as plt
    import numpy as np

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

    # Final plot: total number of events per component per module
    import matplotlib.pyplot as plt

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
    df_mult_fit = (
        pd.concat(
            {k: v for k, v in coeff_tables_normalized.items() if k != 'Total'},
            names=["detection_type", "multiplicity"]
        )
        .reset_index()
        .rename(columns={"level_1": "multiplicity"})
    )

    # Optional: display or save
    # print(df_mult_fit)


print("----------------------------------------------------------------------")
print("------------ Crosstalk probability respect the charge ----------------")
print("----------------------------------------------------------------------")

if crosstalk_probability:
    n_bins = 100
    right_lim = 1400 # 1250
    crosstalk_limit = 2 #2.6
    charge_vector = np.linspace(crosstalk_limit, right_lim, n_bins)

    remove_streamer = False
    streamer_limit = 90

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
    from scipy.interpolate import CubicSpline
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

    # 1. Compute fraction + uncertainty
    x_vals, fraction_hist, frac_err = compute_fraction_and_uncertainty(
        charge_vector,  # bin edges from your snippet
        histograms_no_crosstalk,
        histograms_yes_crosstalk
    )

    # 2. Plot in 4x4
    fig, axs = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(f"Crosstalk probability with Poisson Error Bands, mingo0{station}", fontsize=14)

    for m in range(1, 5):
        for s in range(1, 5):
            ax = axs[m-1, s-1]
            key = f"P{m}_s{s}"

            y_vals = fraction_hist[key]
            y_err  = frac_err[key]

            ax.plot(x_vals, y_vals, label=key)
            ax.fill_between(x_vals, y_vals - y_err, y_vals + y_err, alpha=0.3)
            ax.set_ylim(0, 1)         # Because fraction can be negative if N_yes > N_no
            ax.set_title(key)
            ax.grid(True)

    # Better spacing
    for ax in axs[-1, :]:
        ax.set_xlabel("Charge")
    for ax in axs[:, 0]:
        ax.set_ylabel("Probability")

    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.tight_layout()
    figure_name = f"crosstalk_probability_mingo0{station}"
    if save_plots:
        name_of_file = figure_name
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots: plt.show()
    plt.close()


    # Fit a sigmoidal and store the fitting values to compare with temperature, etc.

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import pandas as pd
    import os

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


print("----------------------------------------------------------------------")
print("------------- cos^n and Georgy's efficiency calculations -------------")
print("----------------------------------------------------------------------")

if georgys:

    # Drop NaN values from theta
    theta_clean = df["theta"].dropna()

    # Define intervals
    in_0_pi2 = ((theta_clean > 0) & (theta_clean < np.pi / 2)).sum()
    in_pi2_pi = ((theta_clean > np.pi / 2) & (theta_clean < np.pi)).sum()
    total = len(theta_clean)

    # Compute percentages
    pct_0_pi2 = 100 * in_0_pi2 / total
    pct_pi2_pi = 100 * in_pi2_pi / total

    print(f"θ ∈ (0, π/2):  {pct_0_pi2:.2f}% of events")
    print(f"θ ∈ (π/2, π):  {pct_pi2_pi:.2f}% of events")

    plt.figure(figsize=(8, 5))
    plt.hist(df["theta"].dropna(), bins=200, alpha=0.7)
    plt.xlabel("Theta (rad)")
    plt.ylabel("Counts")
    plt.title("Histogram of θ")
    plt.grid(True)
    plt.suptitle("Histograms of Cluster and Event Metrics", fontsize=16, y=1.02)
    plt.tight_layout()
    if save_plots:
        name_of_file = 'theta'
        final_filename = f'{fig_idx}_{name_of_file}.png'
        fig_idx += 1
        save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
        plot_list.append(save_fig_path)
        plt.savefig(save_fig_path, format='png')
    if show_plots: plt.show()
    plt.close()

print("\n\n\n")
print(df.columns.to_list())


print("----------------------------------------------------------------------")
print("------------------------- Regions asigning ---------------------------")
print("----------------------------------------------------------------------")

print("Original Region assigning...")
df['region'] = df.apply(classify_region, axis=1)

print("Hans' region assigning...")
df['region_hans'] = df.apply(classify_region_hans, axis=1)

# Clean type column
# print("Cleaning the type column...")
# df['type'] = df['type'].apply(clean_type_column)


print("----------------------------------------------------------------------")
print("----------------- Derived metrics pre-aggregation --------------------")
print("----------------------------------------------------------------------")

# Derived metrics
print("Derived metrics...")
for i in range(1, 5):
    df[f'Q_{i}'] = df[[f'Q_P{i}s{j}' for j in range(1, 5)]].sum(axis=1)
    df[f'count_in_{i}'] = (df[f'Q_{i}'] != 0).astype(int)
    df[f'avalanche_{i}'] = ((df[f'Q_{i}'] != 0) & (df[f'Q_{i}'] < 100)).astype(int)
    df[f'streamer_{i}'] = (df[f'Q_{i}'] > 100).astype(int)


# Streamer percentage
for i in range(1, 5):
    resampled_df[f"streamer_percent_{i}"] = (
        (resampled_df[f"streamer_{i}"] / resampled_df[f"count_in_{i}"])
        .fillna(0) * 100
    )


# ADD THE TOTAL CHARGE OF THE EVENT
for i in range(1, 5):
    df[f'Q_event'] = df[[f'Q_{j}' for j in range(1, 5)]].sum(axis=1)

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


# --- Plot histograms module-wise ---

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


print("----------------------------------------------------------------------")
print("----------------- Aggregation and Poisson filtering ------------------")
print("----------------------------------------------------------------------")

df['events'] = 1

# Aggregation logic
agg_dict = {
    'events': 'sum',
    'x': [custom_mean, custom_std],
    'y': [custom_mean, custom_std],
    'theta': [custom_mean, custom_std],
    'phi': [custom_mean, custom_std],
    # 't0': [custom_mean, custom_std],
    's': [custom_mean, custom_std],
    'type': lambda x: pd.Series(x).value_counts().to_dict(),
    # 'new_type': lambda x: pd.Series(x).value_counts().to_dict(),
    'Q_event': [custom_mean, custom_std],
    "over_P1": "mean",
    "P1-P2": "mean",
    "P2-P3": "mean",
    "P3-P4": "mean",
    "phi_north": "mean",
    "CRT_avg": "mean"
}

for i in range(1, 5):
    agg_dict.update({
        f'count_in_{i}': 'sum',
        f'avalanche_{i}': 'sum',
        f'streamer_{i}': 'sum'
    })

# Fit a Poisson distribution to the 1-second data and removed outliers based on the Poisson -------------------------
if remove_outliers:
    print("Resampling for outlier removal...")
    # Resampling
    resampled_df_test = df.resample('1min', on='Time').agg(agg_dict)
    # TEST 1s rates -------------------------------------------------------
    resampled_second_df = df.resample('1s', on='Time').agg(agg_dict)
    print("Resampled.")
    # Plot the 1 min and 1 s rates ----------------------------------------
    if create_plots:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax[0].plot(resampled_df_test.index, resampled_df_test['events'], label='Q_event')
        ax[0].set_ylabel('Q_event (mean)')
        ax[0].legend()

        ax[1].plot(resampled_second_df.index, resampled_second_df['events'], label='Q_event')
        ax[1].set_ylabel('Q_event (mean)')
        ax[1].legend()

        plt.tight_layout()
        # Show the plot
        if save_plots:
            name_of_file = 'original_rates'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
            
        if show_plots: plt.show()
        plt.close()

    print("Removing outliers...")
    # Data: Replace with your actual data
    data = resampled_second_df['events']

    # Define the negative log-likelihood function for Poisson
    def negative_log_likelihood(lambda_hat, data):
        return -np.sum(poisson.logpmf(data, lambda_hat))

    # Initial guess for λ (mean of data)
    initial_guess = np.mean(data)

    # Optimize λ to minimize the negative log-likelihood
    result = minimize(negative_log_likelihood, x0=initial_guess, args=(data,), bounds=[(1e-5, None)])
    lambda_fit = result.x[0]

    print(f"Best-fit λ: {lambda_fit:.2f}")

    lower_bound = poisson.ppf(0.0005, lambda_fit)  # Lower 00.05% bound
    upper_bound = poisson.ppf(0.9995, lambda_fit)  # Upper 99.95% bound

    # Overlay the fitted Poisson distribution
    if create_plots:
        # Generate Poisson probabilities for the range of your data
        x = np.arange(0, np.max(data) + 1)  # Range of event counts
        poisson_probs = poisson.pmf(x, lambda_fit) * len(data)  # Scale by sample size

        plt.hist(data, bins=100, alpha=0.7, label='Observed data', color='blue', density=False)
        plt.plot(x, poisson_probs, 'r-', lw=2, label=f'Poisson Fit ($\lambda={lambda_fit:.2f}$)')
        plt.axvline(lower_bound, color='r', linestyle='--', label='Poisson 0.1%')
        plt.axvline(upper_bound, color='r', linestyle='--', label='Poisson 99.9%')
        plt.xlabel('Number of events per second')
        plt.ylabel('Frequency')
        plt.title('Histogram of events per second with Poisson Fit')
        plt.legend()
        if save_plots:
            name_of_file = 'poisson_fit'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()
        
    # Now, if any value is outside of the tails of the poisson distribution, we can remove it
    # so obtain the extremes, obtain the seconds in which the values are outside of the distribution
    # and remove those seconds from a copy of the original df, so a new resampled_df can be created
    # with the new data
    
    # Ensure 'events' is a flat Series
    events_series = resampled_second_df['events']
    # Count how many values fall below or above the bounds
    below_count = (events_series < lower_bound).sum()
    above_count = (events_series > upper_bound).sum()
    print("-------------------------------------------------------------")
    print(f"Below bound: {below_count.sum()}")
    print(f"Above bound: {above_count.sum()}")
    print(f"Total outliers: {(below_count + above_count).sum()}")
    print("-------------------------------------------------------------")
    # Identify outlier indices correctly
    outlier_mask = (events_series < lower_bound) | (events_series > upper_bound)
    # Flatten outlier_mask to 1D
    outlier_mask = outlier_mask.values.ravel()  # Turn into a 1D array
    # Extract the indices of outliers
    outlier_indices = events_series.index[outlier_mask]
    # Create a temporary 'Time_sec' column floored to seconds for alignment
    df['Time_sec'] = df['Time'].dt.floor('1s')
    # Filter out rows corresponding to the outlier indices
    filtered_df = df[~df['Time_sec'].isin(outlier_indices)].drop(columns=['Time_sec'])
    # Resample the filtered data to 1-minute intervals
    new_resampled_df = filtered_df.resample('1min', on='Time').agg(agg_dict)

    # Plot the new resampled data
    if create_plots:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(new_resampled_df.index, new_resampled_df['events'], label='Q_event (filtered)')
        ax.set_ylabel('Q_event (mean)')
        ax.set_title('1-Minute Resampled Data After Outlier Removal')
        ax.legend()
        plt.tight_layout()
        if save_plots:
            name_of_file = 'filtered_rate'
            final_filename = f'{fig_idx}_{name_of_file}.png'
            fig_idx += 1
            save_fig_path = os.path.join(base_directories["figure_directory"], final_filename)
            plot_list.append(save_fig_path)
            plt.savefig(save_fig_path, format='png')
        if show_plots: plt.show()
        plt.close()

    # Create a resampled_df with the new one
    
    rejected_percentage = ((below_count + above_count) / len(df)) * 100
    event_acc_global_variables["poisson_rejected"] = rejected_percentage
    
    resampled_df = new_resampled_df
else:
    resampled_df = df.resample('1min', on='Time').agg(agg_dict)


print("----------------------------------------------------------------------")
print("--------------------------- Some renaming ----------------------------")
print("----------------------------------------------------------------------")

resampled_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in resampled_df.columns.values]

rename_map = {
    "over_P1_mean": "over_P1",
    "P1-P2_mean": "P1-P2",
    "P2-P3_mean": "P2-P3",
    "P3-P4_mean": "P3-P4",
    "phi_north_mean": "phi_north"
}
resampled_df.rename(columns=rename_map, inplace=True)

# Replace custom_mean by mean and custom_std by std
resampled_df.rename(columns=lambda x: x.replace('custom_mean', 'mean').replace('custom_std', 'std'), inplace=True)

# Replace sum by nothing
resampled_df.rename(columns=lambda x: x.replace('_sum', ''), inplace=True)


print("----------------------------------------------------------------------")
print("------------------------ Column aggregation --------------------------")
print("----------------------------------------------------------------------")

# Region-specific count aggregation
region_counts = pd.crosstab(df['Time'].dt.floor('1min'), df['region'])
resampled_df = resampled_df.join(region_counts, how='left').fillna(0)

# Hans' region-specific count aggregation
region_hans_counts = pd.crosstab(df['Time'].dt.floor('1min'), df['region_hans'])
resampled_df = resampled_df.join(region_hans_counts, how='left').fillna(0)


print("----------------------------------------------------------------------")
print("-------------------------- Counting types ----------------------------")
print("----------------------------------------------------------------------")

# Split 'type_<lambda>' dictionary into separate columns for each type
if 'type_<lambda>' in resampled_df.columns:
    type_dict_col = resampled_df['type_<lambda>']
    for type_key in df['type'].unique():
        resampled_df[f'type_{type_key}'] = type_dict_col.apply(lambda x: x.get(type_key, 0) if isinstance(x, dict) else 0)
    resampled_df.drop(columns=['type_<lambda>'], inplace=True)


print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")
print("----------------------- Saving and finishing -------------------------")
print("----------------------------------------------------------------------")
print("----------------------------------------------------------------------")

resampled_df.reset_index(inplace=True)

# --- Flatten df_polya_fit ---
df_polya_flat = df_polya_fit.set_index('module').add_prefix('polya_')  # polya_module_1 → polya_theta, etc.

# --- Flatten df_cross_fit ---
df_cross_flat = df_cross_fit.set_index('key').add_prefix('cross_')  # cross_M1_s1_x0, cross_M1_s1_k

# --- Flatten df_mult_fit ---
df_mult_long = df_mult_fit.melt(id_vars=['detection_type', 'multiplicity'], var_name='module', value_name='value')
df_mult_long['colname'] = (
    df_mult_long['detection_type'] + '_' +
    df_mult_long['multiplicity'] + '_' +
    df_mult_long['module']
)
df_mult_flat = df_mult_long.set_index('colname')['value'].to_frame().T.add_prefix('mult_')  # single row

# --- Combine all into one row DataFrame ---
flat_polya = df_polya_flat.stack().to_frame().T
flat_cross = df_cross_flat.stack().to_frame().T
flat_mult = df_mult_flat

flat_all = pd.concat([flat_polya, flat_cross, flat_mult], axis=1)

# --- Repeat and merge with resampled_df ---
flat_all_repeated = pd.concat([flat_all] * len(resampled_df), ignore_index=True)
resampled_df = pd.concat([resampled_df.reset_index(drop=True), flat_all_repeated], axis=1)


print("----------------------------------------------------------------------")
print("------------------------- Saving the data ----------------------------")
print("----------------------------------------------------------------------")

# Print the columns of resampled_df
print("\n\n")
print(resampled_df.columns.to_list())
print("\n\n")

resampled_df = resampled_df.applymap(round_to_significant_digits)

# Save the newly created file to ACC_EVENTS_DIRECTORY --------------------------
resampled_df.to_csv(full_save_path, sep=',', index=False)
print(f"Complete datafile saved in {full_save_filename}. Path is {full_save_path}")

columns_to_keep = [
    # Timestamp and identifiers
    'Time'
    # 'Time', 'original_tt', 'processed_tt', 'tracking_tt',

    # # Summary metrics and quality flags
    # 'CRT_avg', 'sigmoid_width', 'background_slope',
    # 'one_side_events', 'purity_of_data_percentage',
    # 'unc_y', 'unc_tsum', 'unc_tdif',

    # # Alternative reconstruction outputs
    # 'alt_x', 'alt_y', 'alt_theta', 'alt_phi', 'alt_s', 'alt_th_chi',

    # # TimTrack reconstruction outputs
    # 'x', 'y', 'theta', 'phi', 's', 'th_chi',

    # # Strip-level time and charge info (ordered by plane and strip)
    # *[f'Q_P{p}s{s}' for p in range(1, 5) for s in range(1, 5)]
]

reduced_df = resampled_df[columns_to_keep]
reduced_df.to_csv(save_path, index=False, sep=',', float_format='%.5g')
print(f"Reduced columns datafile saved in {save_filename}. Path is {save_path}")

# Move the original file in file_path to completed_directory
print("Moving file to COMPLETED directory...")
shutil.move(file_path, completed_file_path)
print(f"File moved to: {completed_file_path}")


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