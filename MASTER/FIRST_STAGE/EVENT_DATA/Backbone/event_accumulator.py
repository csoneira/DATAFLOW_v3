#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
print("--------- Running event_accumulator.py -------------------------------")
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

remove_outliers = True
create_plots = True
save_plots = True
create_pdf = True
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
# -----------------------------------------------------------------------------

# Define input file path -----------------------------------------------------
input_file_config_path = os.path.join(station_directory, f"input_file_mingo0{station}.csv")

if os.path.exists(input_file_config_path):
    # It is a csv
    input_file = pd.read_csv(input_file_config_path, skiprows=1)
    
    print("Input configuration file found.")
    exists_input_file = True
    
    # Print the head
    # print(input_file.head())
    
else:
    exists_input_file = False
    print("Input configuration file does not exist.")
    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Configurations --------------------------------------------------------------
# -----------------------------------------------------------------------------

force_replacement = True  # Creates a new datafile even if there is already one that looks complete
show_plots = False
high_mid_limit_angle = 15

# crosstalk_threshold = 1.2

regions = ['High', 'N', 'E', 'S', 'W']
# test_filename = 'list_events_2024.12.16_23.27.54.txt'


# -----------------------------------------------------------------------------
# Functions -------------------------------------------------------------------
# -----------------------------------------------------------------------------

def classify_region(row):
    phi = row['phi'] * 180 / np.pi  + row['phi_north'] # Convert phi to degrees
    theta = row['theta'] * 180 / np.pi
    if 0 <= theta < high_mid_limit_angle:
        return 'High'
    elif high_mid_limit_angle <= theta <= 90:
        if -45 <= phi < 45:
            return 'N'
        elif 45 <= phi < 135:
            return 'E'
        elif -135 <= phi < -45:
            return 'W'
        else:
            return 'S'

def custom_mean(x):
    return x[x != 0].mean() if len(x[x != 0]) > 0 else 0

def custom_std(x):
    return x[x != 0].std() if len(x[x != 0]) > 0 else 0

def round_to_significant_digits(x):
    if isinstance(x, float):
        return float(f"{x:.6g}")
    return x

def clean_type_column(x):
    return str(int(float(x))) if isinstance(x, (float, int, str)) and not pd.isna(x) else x

# -----------------------------------------------------------------------------
# Main Script -----------------------------------------------------------------
# -----------------------------------------------------------------------------

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


# Input file
df = pd.read_csv(file_path, sep=',')

# Data preparation
df['Time'] = pd.to_datetime(df['Time'], errors='coerce') # Added errors='coerce' to handle NaT values

# Print the number of events (rows) in the file
print(f"Number of events in the file: {len(df)}")

# Get the minimum value directly from the column (may include NaT)
min_time_original = df['Time'].min()
max_time_original = df['Time'].max()

# Filter out invalid or null datetime values
valid_times = df['Time'].dropna()

# Get the smallest valid datetime
if not valid_times.empty:
    min_time_valid = valid_times.min()
    
    # Check if the min value with NaT differs from the valid min value
    if min_time_original != min_time_valid:
        print("Notice: The minimum value from 'Time' column differs from the smallest valid datetime.")
        print("Original min value (including NaT):", min_time_original)
        print("Valid min value (ignoring NaT):", min_time_valid)
    
    # Use the valid min datetime
    first_datetime = min_time_valid
    
    # Define filename save suffix in the format 'yy-mm-dd_HH.MM.SS'
    filename_save_suffix = first_datetime.strftime('%y-%m-%d_%H.%M.%S')
else:
    # first_datetime = None
    sys.exit("No valid datetime values found in the 'Time' column. Exiting...")

print("Filename save suffix:", filename_save_suffix)

save_filename = f"accumulated_events_{filename_save_suffix}.csv"
save_path = os.path.join(base_directories["acc_events_directory"], save_filename)

save_pdf_filename = f"pdf_{filename_save_suffix}.pdf"
save_pdf_path = os.path.join(base_directories["pdf_directory"], save_pdf_filename)


# ---------------------------------------------------------------------------------------------------------------------
# Input file reading --------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

if exists_input_file:
    start_time = min_time_original
    end_time = max_time_original

    # Print types of start and end dates
    # print(f"Start date type: {type(start_time)}") # Start date type: <class 'pandas._libs.tslibs.timestamps.Timestamp'>
    # print(f"End date type: {type(end_time)}") # End date type: <class 'pandas._libs.tslibs.timestamps.Timestamp'>

    input_file["start"] = pd.to_datetime(input_file["start"])
    input_file["end"] = pd.to_datetime(input_file["end"])

    # Ensure no NaN in 'end' column
    input_file["end"].fillna(pd.to_datetime('now'), inplace=True)
    
    matching_confs = input_file[ (input_file["start"] <= start_time) & (input_file["end"] >= end_time) ]
    
    if not matching_confs.empty:
    
        if len(matching_confs) > 1:
            print(f"Warning: Multiple configurations match the date range ({start_time} to {end_time}).")
                
            # Create an empty dictionary to hold new column values
            new_columns = {
                "over_P1": [],
                "P1-P2": [],
                "P2-P3": [],
                "P3-P4": [],
                "phi_north": []
            }

            # Print df columns
            print(df.columns)

            # Assign values based on corresponding time range
            for timestamp in df["Time"]:
                # Find matching configuration
                match = input_file[
                    (input_file["start"] <= timestamp) & (input_file["end"] >= timestamp)
                ]
                
                if not match.empty:
                    # Take the first matching row
                    selected_conf = match.iloc[0]
                    new_columns["over_P1"].append(selected_conf.get("over_P1", np.nan))
                    new_columns["P1-P2"].append(selected_conf.get("P1-P2", np.nan))
                    new_columns["P2-P3"].append(selected_conf.get("P2-P3", np.nan))
                    new_columns["P3-P4"].append(selected_conf.get("P3-P4", np.nan))
                    new_columns["phi_north"].append(selected_conf.get("phi_north", np.nan))
                else:
                    # No matching configuration, fill with NaN
                    new_columns["over_P1"].append(np.nan)
                    new_columns["P1-P2"].append(np.nan)
                    new_columns["P2-P3"].append(np.nan)
                    new_columns["P3-P4"].append(np.nan)
                    new_columns["phi_north"].append(0)  # Default value for phi_north

            # Convert dictionary to DataFrame
            df_new_cols = pd.DataFrame(new_columns)

            # Merge with the original dataframe
            df_extended = pd.concat([df, df_new_cols], axis=1)

            # Fill missing values with the original values where applicable
            df_extended.fillna(method='ffill', inplace=True)

            # Print the new columns
            df = df_extended

            # print(df.columns)
            # print(df.head())
            # print(df.tail())
            
        if len(matching_confs) == 1:
            selected_conf = matching_confs.iloc[0]
            print(f"Only one selected configuration: {selected_conf['conf']}")
            
            df["over_P1"] = selected_conf.get("over_P1", np.nan)
            df["P1-P2"] = selected_conf.get("P1-P2", np.nan)
            df["P2-P3"] = selected_conf.get("P2-P3", np.nan)
            df["P3-P4"] = selected_conf.get("P3-P4", np.nan)
            df["phi_north"] = selected_conf.get("phi_north", 0)

    else:
        # Create new columns with default values
        df["over_P1"] = 0
        df["P1-P2"] = 0
        df["P2-P3"] = 0
        df["P3-P4"] = 0
        df["phi_north"] = 0

else:
    # Create new columns with default values
    df["over_P1"] = 0
    df["P1-P2"] = 0
    df["P2-P3"] = 0
    df["P3-P4"] = 0
    df["phi_north"] = 0
    
# ---------------------------------------------------------------------------------------------------------------------

# Start the analysis --------------------------------------------------------

print("Starting the analysis")

print("Region assigning...")
df['region'] = df.apply(classify_region, axis=1)

# Clean type column
print("Cleaning the type column...")
df['type'] = df['type'].apply(clean_type_column)

# Derived metrics
print("Derived metrics...")
for i in range(1, 5):
    df[f'Q_{i}'] = df[[f'Q_M{i}s{j}' for j in range(1, 5)]].sum(axis=1)
    df[f'count_in_{i}'] = (df[f'Q_{i}'] != 0).astype(int)
    df[f'avalanche_{i}'] = ((df[f'Q_{i}'] != 0) & (df[f'Q_{i}'] < 100)).astype(int)
    df[f'streamer_{i}'] = (df[f'Q_{i}'] > 100).astype(int)

# ADD THE TOTAL CHARGE OF THE EVENT
for i in range(1, 5):
    df[f'Q_event'] = df[[f'Q_{j}' for j in range(1, 5)]].sum(axis=1)

# Make a column which is the a string with the number of planes that have a charge that is not streamer
# column = []
# for row in df.iterrows():
#     new_type = ""
#     for i in range(1, 5):
#         if row[1][f'Q_{i}'] > 0 and row[1][f'Q_{i}'] < 100:
#             new_type += str(i)
#     column.append(new_type)
# df['new_type'] = column

# df['new_type'] = df.apply(lambda row: [i for i in range(1, 5) if 0 < row[f'Q_{i}'] < 100], axis=1)
# print(df['new_type'])

# Work for the future ------------------------------------------------------------------------------
# df['topology_0'] = df.apply(lambda row: sum(row[f'Q_{i}'] > 0 for i in range(1, 5)), axis=1)
# df[f'topology_{crosstalk_threshold}'] = df.apply(lambda row: sum(row[f'Q_{i}'] > crosstalk_threshold for i in range(1, 5)), axis=1)

df['events'] = 1

# Aggregation logic
agg_dict = {
    'events': 'sum',
    'x': [custom_mean, custom_std],
    'y': [custom_mean, custom_std],
    'theta': [custom_mean, custom_std],
    'phi': [custom_mean, custom_std],
    't0': [custom_mean, custom_std],
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

# Fit a Poisson distribution to the 1-second data and removed outliers based on the Poisson -------------------------
if remove_outliers:
    
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

        # Plot histogram of the data
        plt.hist(data, bins=100, alpha=0.7, label='Observed data', color='blue', density=False)

        plt.plot(x, poisson_probs, 'r-', lw=2, label=f'Poisson Fit ($\lambda={lambda_fit:.2f}$)')
        plt.axvline(lower_bound, color='r', linestyle='--', label='Poisson 0.1%')
        plt.axvline(upper_bound, color='r', linestyle='--', label='Poisson 99.9%')

        # Add labels, legend, and title
        plt.xlabel('Number of events per second')
        plt.ylabel('Frequency')
        plt.title('Histogram of events per second with Poisson Fit')
        plt.legend()

        # Show the plot
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
        
        # Show the plot
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
    resampled_df = new_resampled_df

# ----------------------------------------------------------------------

resampled_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in resampled_df.columns.values]

rename_map = {
    "over_P1_mean": "over_P1",
    "P1-P2_mean": "P1-P2",
    "P2-P3_mean": "P2-P3",
    "P3-P4_mean": "P3-P4",
    "phi_north_mean": "phi_north"
}
new_resampled_df.rename(columns=rename_map, inplace=True)

# Replace custom_mean by mean and custom_std by std
resampled_df.rename(columns=lambda x: x.replace('custom_mean', 'mean').replace('custom_std', 'std'), inplace=True)

# Replace sum by nothing
resampled_df.rename(columns=lambda x: x.replace('_sum', ''), inplace=True)

# Region-specific count aggregation
region_counts = pd.crosstab(df['Time'].dt.floor('1min'), df['region'])
resampled_df = resampled_df.join(region_counts, how='left').fillna(0)

# Split 'type_<lambda>' dictionary into separate columns for each type
if 'type_<lambda>' in resampled_df.columns:
    type_dict_col = resampled_df['type_<lambda>']
    for type_key in df['type'].unique():
        resampled_df[f'type_{type_key}'] = type_dict_col.apply(lambda x: x.get(type_key, 0) if isinstance(x, dict) else 0)
    resampled_df.drop(columns=['type_<lambda>'], inplace=True)

# print("Test--------------------------------------------")
# if 'new_type_<lambda>' in resampled_df.columns:
#     type_dict_col = resampled_df['new_type_<lambda>']
#     for type_key in df['new_type'].unique():
#         # print(type_key)
#         resampled_df[f'new_type_{type_key}'] = type_dict_col.apply(lambda x: x.get(type_key, 0) if isinstance(x, dict) else 0)
#     resampled_df.drop(columns=['new_type_<lambda>'], inplace=True)

# Streamer percentage
for i in range(1, 5):
    resampled_df[f"streamer_percent_{i}"] = (
        (resampled_df[f"streamer_{i}"] / resampled_df[f"count_in_{i}"])
        .fillna(0) * 100
    )


# -----------------------------------------------------------------------------
# Saving the file -------------------------------------------------------------
# -----------------------------------------------------------------------------

# Show the head and tail
print(resampled_df.head())
print(resampled_df.tail())

# Print the columns
print(resampled_df.columns)

resampled_df.reset_index(inplace=True)
resampled_df = resampled_df.applymap(round_to_significant_digits)

# Save the newly created file to ACC_EVENTS_DIRECTORY --------------------------
resampled_df.to_csv(save_path, sep=',', index=False)
print(f"Saved data to file: {save_path}")

# Move the original file in file_path to completed_directory
print("Moving file to COMPLETED directory...")
shutil.move(file_path, completed_file_path)
print(f"File moved to: {completed_file_path}")


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


