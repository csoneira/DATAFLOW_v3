#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os
import builtins
import shutil


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


# -----------------------------------------------------------------------------

working_directory = os.path.expanduser(f"~/DATAFLOW_v3/STATIONS/MINGO0{station}/FIRST_STAGE/EVENT_DATA")
acc_working_directory = os.path.join(working_directory, "LIST_TO_ACC")

# Define subdirectories relative to the working directory
base_directories = {
    "list_events_directory": os.path.join(working_directory, "LIST_EVENTS_DIRECTORY"),
    
    "unprocessed_directory": os.path.join(acc_working_directory, "ACC_FILES/ACC_UNPROCESSED"),
    "processing_directory": os.path.join(acc_working_directory, "ACC_FILES/ACC_PROCESSING"),
    "completed_directory": os.path.join(acc_working_directory, "ACC_FILES/ACC_COMPLETED"),
    
    "list_events_directory": os.path.join(working_directory, "LIST_EVENTS_DIRECTORY"),
}

# Create ALL directories if they don't already exist
for directory in base_directories.values():
    os.makedirs(directory, exist_ok=True)

# Path to big_event_data.csv
big_event_file = os.path.join(working_directory, "big_event_data.csv")


# Move files from RAW to RAW_TO_LIST/RAW_TO_LIST_FILES/UNPROCESSED,
# ensuring that only files not already in UNPROCESSED, PROCESSING,
# or COMPLETED are moved:

list_events_directory = base_directories["list_events_directory"]
unprocessed_directory = base_directories["unprocessed_directory"]
processing_directory = base_directories["processing_directory"]
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
        shutil.move(src_path, dest_path)
        print(f"Move {file_name} to UNPROCESSED directory.")
    except Exception as e:
        print(f"Failed to copy {file_name}: {e}")



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Configurations --------------------------------------------------------------
# -----------------------------------------------------------------------------

show_plots = False
force_replacement = True  # Creates a new datafile even if there is already one that looks complete
high_mid_limit_angle = 15

crosstalk_threshold = 1.2

regions = ['High', 'N', 'E', 'S', 'W']
test_filename = 'list_events_2024.12.16_23.27.54.txt'


# -----------------------------------------------------------------------------
# Functions -------------------------------------------------------------------
# -----------------------------------------------------------------------------

def classify_region(row):
    phi = row['phi'] * 180 / np.pi  # Convert phi to degrees
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

def round_to_4_significant_digits(x):
    if isinstance(x, float):
        return float(f"{x:.4g}")
    return x

def clean_type_column(x):
    return str(int(float(x))) if isinstance(x, (float, int, str)) and not pd.isna(x) else x

# -----------------------------------------------------------------------------
# Main Script -----------------------------------------------------------------
# -----------------------------------------------------------------------------

# Create ALL directories if they don't already exist
for directory in base_directories.values():
    os.makedirs(directory, exist_ok=True)

# # Determine the file path input
# try:
#     file_path_input = sys.argv[1]
#     print("Running with given input.")
    
#     file_name = os.path.basename(file_path_input)  # Extract just the file name
    
#     processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
#     completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

#     # Check if the file is already in PROCESSING
#     if os.path.exists(processing_file_path):
#         print(f"File '{file_name}' is already in PROCESSING. Continuing with processing...")
#     else:
#         # Check if the file exists at the input location before moving
#         if os.path.exists(file_path_input):
#             print(f"Moving input file '{file_name}' to PROCESSING directory...")
#             shutil.move(file_path_input, processing_file_path)
#             print(f"File moved to PROCESSING directory: {processing_file_path}")
#         else:
#             raise FileNotFoundError(f"Input file '{file_path_input}' not found and not in PROCESSING.")
    
#     file_path = processing_file_path
    
# except IndexError:
#     file_path = test_filename
#     print(f"--> Reading test filename: '{file_path}'")



unprocessed_files = os.listdir(base_directories["unprocessed_directory"])
if unprocessed_files:
    for file_name in unprocessed_files:
        unprocessed_file_path = os.path.join(base_directories["unprocessed_directory"], file_name)
        processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
        completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

        # Skip if file is already in COMPLETED
        if os.path.exists(completed_file_path):
            print(f"File '{file_name}' is already in COMPLETED. Removing from UNPROCESSED...")
            os.remove(unprocessed_file_path)
            continue

        # Skip if file is already in PROCESSING
        if os.path.exists(processing_file_path):
            print(f"File '{file_name}' is already in PROCESSING. Removing from UNPROCESSED...")
            os.remove(unprocessed_file_path)
            continue

        # Move file to PROCESSING and process it
        print(f"Moving '{file_name}' to PROCESSING...")
        shutil.move(unprocessed_file_path, processing_file_path)
        print(f"File moved to PROCESSING: {processing_file_path}")
else:
    # Check for files in PROCESSING
    processing_files = os.listdir(base_directories["processing_directory"])
    if processing_files:
        for file_name in processing_files:
            processing_file_path = os.path.join(base_directories["processing_directory"], file_name)
            completed_file_path = os.path.join(base_directories["completed_directory"], file_name)

            # If already in COMPLETED, remove it from PROCESSING
            if os.path.exists(completed_file_path):
                print(f"File '{file_name}' is already in COMPLETED. Removing from PROCESSING...")
                os.remove(processing_file_path)
                continue

            # Otherwise, process the file
            print(f"Processing file in PROCESSING: {file_name}")
    else:
        sys.exit("No files to process in UNPROCESSED or PROCESSING.")

file_path = processing_file_path

# Input file
df = pd.read_csv(file_path, sep=',')

# Data preparation
df['Time'] = pd.to_datetime(df['Time'], errors='coerce') # Added errors='coerce' to handle NaT values
df['region'] = df.apply(classify_region, axis=1)

# Get the minimum value directly from the column (may include NaT)
min_time_original = df['Time'].min()

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
    print("Error: No valid datetime values found in the 'Time' column.")
    first_datetime = None
    exit(1)  # Exit the program

print("Filename save suffix:", filename_save_suffix)


# Clean type column
df['type'] = df['type'].apply(clean_type_column)

# Derived metrics
for i in range(1, 5):
    df[f'Q_{i}'] = df[[f'Q_M{i}s{j}' for j in range(1, 5)]].sum(axis=1)
    df[f'count_in_{i}'] = (df[f'Q_{i}'] > 0).astype(int)
    df[f'avalanche_{i}'] = ((df[f'Q_{i}'] > 0) & (df[f'Q_{i}'] < 100)).astype(int)
    df[f'streamer_{i}'] = (df[f'Q_{i}'] > 100).astype(int)

# ADD THE TOTAL CHARGE OF THE EVENT
for i in range(1, 5):
    df[f'Q_event'] = df[[f'Q_{j}' for j in range(1, 5)]].sum(axis=1)


# Work for the future ------------------------------------------------------------------------------
# df['topology_0'] = df.apply(lambda row: sum(row[f'Q_{i}'] > 0 for i in range(1, 5)), axis=1)
# df[f'topology_{crosstalk_threshold}'] = df.apply(lambda row: sum(row[f'Q_{i}'] > crosstalk_threshold for i in range(1, 5)), axis=1)


# Aggregation logic
agg_dict = {
    'x': [custom_mean, custom_std],
    'y': [custom_mean, custom_std],
    'theta': [custom_mean, custom_std],
    'phi': [custom_mean, custom_std],
    't0': [custom_mean, custom_std],
    's': [custom_mean, custom_std],
    'type': lambda x: pd.Series(x).value_counts().to_dict(),
    'Q_event': [custom_mean, custom_std],
}

for i in range(1, 5):
    agg_dict.update({
        f'count_in_{i}': 'sum',
        f'avalanche_{i}': 'sum',
        f'streamer_{i}': 'sum'
    })

# Resampling
resampled_df = df.resample('1min', on='Time').agg(agg_dict)
resampled_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in resampled_df.columns.values]

# Region-specific count aggregation
region_counts = pd.crosstab(df['Time'].dt.floor('1min'), df['region'])
resampled_df = resampled_df.join(region_counts, how='left').fillna(0)

# Split 'type_<lambda>' dictionary into separate columns for each type
if 'type_<lambda>' in resampled_df.columns:
    type_dict_col = resampled_df['type_<lambda>']
    for type_key in df['type'].unique():
        resampled_df[f'type_{type_key}'] = type_dict_col.apply(lambda x: x.get(type_key, 0) if isinstance(x, dict) else 0)
    resampled_df.drop(columns=['type_<lambda>'], inplace=True)



# -----------------------------------------------------------------------------
# Saving the file -------------------------------------------------------------
# -----------------------------------------------------------------------------

resampled_df.reset_index(inplace=True)

# save_filename = f"accumulated_events_{filename_save_suffix}.csv"

resampled_df = resampled_df.applymap(round_to_4_significant_digits)

# resampled_df.to_csv(save_filename, sep=',', index=False)
# print(f"Saved data to file: {save_filename}")


# -----------------------------------------------------------------------------
# Merge with big_event_data.csv and handle duplicates -------------------------
# -----------------------------------------------------------------------------

# Load or create big_event_data.csv
if os.path.exists(big_event_file):
    big_event_df = pd.read_csv(big_event_file, sep=',', parse_dates=['Time'])
    print(f"Loaded existing big_event_data.csv with {len(big_event_df)} rows.")
else:
    big_event_df = pd.DataFrame(columns=resampled_df.columns)
    print("Created new empty big_event_data.csv dataframe.")

# Concatenate the new resampled data with the existing big_event_data
combined_df = pd.concat([big_event_df, resampled_df], ignore_index=True)

# Function to handle duplicates in 'Time'
def combine_duplicates(group):
    if len(group) == 1:
        return group.iloc[0]  # No duplicates to combine
    
    # Check if all rows are identical (excluding 'Time')
    rows_identical = group.drop(columns='Time').nunique().sum() == 0
    
    if rows_identical:
        return group.iloc[0]  # Keep any one row if all values are identical
    
    # Columns to sum and average
    sum_columns = [col for col in group.columns if any(prefix in col for prefix in 
                    ['count_in_', 'avalanche_', 'streamer_', 'High', 'N', 'E', 'S', 'W', 'type_'])]
    avg_columns = ['x_custom_mean', 'y_custom_mean', 'theta_custom_mean', 'phi_custom_mean', 
                   't0_custom_mean', 's_custom_mean', 'Q_event_custom_mean']

    result = {}
    for col in group.columns:
        if col in sum_columns:
            result[col] = group[col].sum()  # Sum values
        elif col in avg_columns:
            result[col] = group[col].mean()  # Average values
        elif col.endswith('_custom_std'):
            result[col] = group[col].mean()  # Average standard deviations
        else:
            result[col] = group[col].iloc[0]  # Take any non-aggregated value

    return pd.Series(result)

# Group by 'Time' to combine duplicates
combined_df = combined_df.groupby('Time', as_index=False).apply(combine_duplicates)

# Sort the combined DataFrame by 'Time'
combined_df = combined_df.sort_values(by='Time').reset_index(drop=True)

# -----------------------------------------------------------------------------
# Save the updated big_event_data.csv -----------------------------------------
# -----------------------------------------------------------------------------
combined_df = combined_df.applymap(round_to_4_significant_digits)
combined_df.to_csv(big_event_file, sep=',', index=False)
print(f"Updated big_event_data.csv with {len(combined_df)} rows.")

# Move the original datafile to PROCESSED -------------------------------------
print("Moving file to COMPLETED directory...")
shutil.move(processing_file_path, completed_file_path)
print(f"File moved to: {completed_file_path}")